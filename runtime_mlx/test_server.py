"""Tests for `lamq serve` (runtime_mlx/server.py) that need no model weights."""
from __future__ import annotations

import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_mlx.server import SPECIAL, OutputParser, RequestError, make_handler, normalize_messages  # noqa: E402

IDS = {k: 1000 + i for i, k in enumerate(SPECIAL)}
NAMES = {v: SPECIAL[k] for k, v in IDS.items()}


class ByteTokenizer:
    """Ids < 256 are raw bytes (so a multi-byte character can split across tokens); 1000+ are specials."""

    def encode(self, s: str) -> list[int]:
        out, i = [], 0
        while i < len(s):
            for k, name in SPECIAL.items():
                if s.startswith(name, i):
                    out.append(IDS[k])
                    i += len(name)
                    break
            else:
                out.extend(s[i].encode())
                i += 1
        return out

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) -> str:
        out, buf = [], b""
        for t in ids:
            if t in NAMES:
                out.append(buf.decode(errors="replace") + NAMES[t])
                buf = b""
            else:
                buf += bytes([t])
        return "".join(out) + buf.decode(errors="replace")


def parse(text: str):
    tok = ByteTokenizer()
    p = OutputParser(tok, IDS)
    events = []
    for t in tok.encode(text):
        events += p.feed(t)
        if p.done:
            break
    events += p.close()
    reasoning = "".join(e[1] for e in events if e[0] == "reasoning")
    content = "".join(e[1] for e in events if e[0] == "content")
    tools = [e[1]["function"] for e in events if e[0] == "tool"]
    return reasoning, content, tools, p.finish


def test_output_parser() -> None:
    r, c, t, f = parse("<|message_model|><|content_thinking|>First.<|end_message|><|message_model|><|content_thinking|>Second."
                       "<|end_message|><|message_model|><|content_text|>café ☕ 日本 🎉<|end_message|><|content_model_end_sampling|>")
    assert (r, c, t, f) == ("First.\n\nSecond.", "café ☕ 日本 🎉", [], "stop"), (r, c, t, f)

    r, c, t, f = parse('<|message_model|><|content_thinking|>Read it.<|end_message|><|message_model|>read<|content_invoke_tool_json|>'
                       '{"name":"read","args":{"path":"a b/ü.py"}}<|end_message|><|message_model|>bash<|content_invoke_tool_json|>{}')
    assert r == "Read it." and c == "" and f == "tool_calls"
    assert t == [{"name": "read", "arguments": '{"path": "a b/ü.py"}'}], "stops after the first tool call"

    assert parse("bare text, no header<|content_model_end_sampling|>")[1] == "bare text, no header"
    assert parse("<|message_model|><|content_text|>hi<|end_message|><|message_user|>imagined turn")[1] == "hi"
    _, _, t, _ = parse("<|message_model|>bash<|content_invoke_tool_json|>not json<|end_message|>")
    assert t == [{"name": "bash", "arguments": "not json"}], "unparseable arguments pass through raw"


def test_normalize_messages() -> None:
    out = normalize_messages([
        {"role": "developer", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None, "reasoning_content": "think",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "x"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "data"},
    ])
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1]["content"] == "ab"
    assert out[2]["content"] == "" and out[2]["reasoning_content"] == "think"
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"path": "x"}
    assert out[3] == {"role": "tool", "content": "data", "tool_call_id": "c1"}
    for bad in ([], [{"role": "wizard", "content": "x"}]):
        try:
            normalize_messages(bad)
            raise AssertionError("expected RequestError")
        except RequestError as e:
            assert e.status == 400


class FakeEngine:
    args = SimpleNamespace(context=4096)

    def prepare(self, req):
        if req.get("messages") == "too long":
            raise RequestError(400, "the request exceeds the available context size, try increasing it", "exceed_context_size_error")
        return {"want_tool": bool(req.get("tools"))}

    def run(self, job, emit, gone):
        emit("reasoning", "thinking")
        if job["want_tool"]:
            emit("tool", {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path": "x"}'}})
            finish = "tool_calls"
        else:
            emit("content", "Hello")
            emit("content", " world")
            finish = "stop"
        return {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, "prompt_tokens_details": {"cached_tokens": 4}}, finish


def test_http_round_trip() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), "inkling-test"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def call(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, r.getheader("Content-Type"), data

    try:
        assert json.loads(call("GET", "/health")[2]) == {"status": "ok"}
        models = json.loads(call("GET", "/v1/models")[2])
        assert models["data"][0]["id"] == "inkling-test" and models["data"][0]["context_length"] == 4096
        assert call("GET", "/nope")[0] == 404

        st, _, data = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        r = json.loads(data)
        msg = r["choices"][0]["message"]
        assert st == 200 and msg["content"] == "Hello world" and msg["reasoning_content"] == "thinking"
        assert r["choices"][0]["finish_reason"] == "stop" and r["usage"]["prompt_tokens_details"]["cached_tokens"] == 4

        st, ctype, data = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "tools": [{}],
                                                                "stream": True, "stream_options": {"include_usage": True}})
        assert st == 200 and ctype == "text/event-stream"
        lines = [ln[6:] for ln in data.decode().split("\n") if ln.startswith("data: ")]
        assert lines[-1] == "[DONE]"
        events = [json.loads(x) for x in lines[:-1]]
        deltas = [e["choices"][0]["delta"] for e in events if e["choices"]]
        assert deltas[0] == {"role": "assistant", "content": ""}
        assert {"reasoning_content": "thinking"} in deltas
        tc = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
        assert tc == [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path": "x"}'}, "index": 0}]
        assert [e["choices"][0]["finish_reason"] for e in events if e["choices"]][-1] == "tool_calls"
        assert events[-1]["choices"] == [] and events[-1]["usage"]["total_tokens"] == 13

        st, _, data = call("POST", "/v1/chat/completions", {"messages": "too long", "stream": True})
        err = json.loads(data)["error"]
        assert st == 400 and err["type"] == "exceed_context_size_error" and "exceeds the available context size" in err["message"]
    finally:
        srv.shutdown()
        srv.server_close()


def main() -> None:
    test_output_parser()
    test_normalize_messages()
    test_http_round_trip()
    print("ALL LAMQ SERVE TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
