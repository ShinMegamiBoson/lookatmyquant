"""Tests for `lamq serve` (runtime_mlx/server.py) that need no model weights."""
from __future__ import annotations

import http.client
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_mlx.server import (
    MAX_REQUEST_BYTES, SPECIAL, BoundedHTTPServer, Engine, OutputParser, RequestError, make_handler, normalize_messages,
)  # noqa: E402

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


def parse(text: str, tool_names=()):
    tok = ByteTokenizer()
    p = OutputParser(tok, IDS, tool_names)
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
                       '{"name":"read","args":{"path":"a b/ü.py"}}<|end_message|><|message_model|>bash<|content_invoke_tool_json|>{}', {"read"})
    assert r == "Read it." and c == "" and f == "tool_calls"
    assert t == [{"name": "read", "arguments": '{"path": "a b/ü.py"}'}], "stops after the first tool call"

    assert parse("bare text, no header<|content_model_end_sampling|>")[1] == "bare text, no header"
    assert parse("<|message_model|><|content_text|>hi<|end_message|><|message_user|>imagined turn")[1] == "hi"


def test_tool_calls_require_a_complete_declared_json_frame() -> None:
    model, tool, end, eos = (SPECIAL[k] for k in ("model", "tool_json", "end", "eos"))
    text = model + SPECIAL["text"] + "Continuing normally." + end + eos
    declared = {"read", "list"}
    for header, body in (
        ("read", '{"path":"x"}'),
        ("read", '{"name":"read","args":{"path":"x"}}'),
        ("read", '{"arguments":{"path":"x"}}'),
        ("", '{"name":"read","arguments":{"path":"x"}}'),
    ):
        r, c, calls, finish = parse(model + header + tool + body + end, declared)
        assert (r, c, calls, finish) == ("", "", [{"name": "read", "arguments": '{"path": "x"}'}], "tool_calls")

    for header, body, marker, names in (
        ("read", '{"path":"x"}', tool, set()),
        ("other", '{"path":"x"}', tool, declared),
        ("read", '{"name":"list","args":{}}', tool, declared),
        ("read", "not JSON", tool, declared),
        ("read", '{"name":"read","args":{}', tool, declared),
        ("read", '["x"]', tool, declared),
        ("read", '{"name":"read","args":null}', tool, declared),
        ("read", '{"name":"read","args":[]}', tool, declared),
        ("read", '{"value":NaN}', tool, declared),
        ("read", '{"value":1e400}', tool, declared),
        ("read", r'{"value":"\ud800"}', tool, declared),
        ("read", '{"path":"x"}', SPECIAL["tool_text"], declared),
    ):
        assert parse(model + header + marker + body + end + text, names) == ("", "Continuing normally.", [], "stop")
    incomplete = model + "read" + tool + '{"path":"x"}'
    assert parse(incomplete, declared)[2:] == ([], "stop")
    assert parse(incomplete + eos, declared)[2:] == ([], "stop")
    assert parse(incomplete + text, declared)[1:] == ("Continuing normally.", [], "stop")


def test_prepare_carries_only_declared_tool_names() -> None:
    engine = Engine.__new__(Engine)
    engine.args = SimpleNamespace(context=64, max_tokens=16, reasoning_effort="none", temp=0.0, top_p=1.0)
    engine.tok = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2])
    engine.m = SimpleNamespace(lm_head=SimpleNamespace(order_mx=SimpleNamespace(size=1024)))
    req = {"messages": [{"role": "user", "content": "Hello."}]}
    assert engine.prepare(req)["tool_names"] == set()
    assert engine.prepare(dict(req, tools=[{"type": "function", "function": {"name": "read"}}]))["tool_names"] == {"read"}
    for tools in ({}, [{}], [{"type": "function", "function": {"name": []}}]):
        try:
            engine.prepare(dict(req, tools=tools))
        except RequestError as e:
            assert e.status == 400
        else:
            raise AssertionError("invalid tool declaration")


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

    def render(self, req):
        return req["messages"][0]["content"]

    def tokenize(self, content, add_special):
        return list(content.encode()) + ([999] if add_special else [])

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
    srv = BoundedHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), "inkling-test"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def call(method, path, body=None, headers=None):
        return _call(port, method, path, body, headers)

    try:
        assert json.loads(call("GET", "/health")[2]) == {"status": "ok"}
        models = json.loads(call("GET", "/v1/models")[2])
        assert models["data"][0]["id"] == "inkling-test" and models["data"][0]["context_length"] == 4096
        assert call("GET", "/nope")[0] == 404
        assert call("GET", "/v1/models", headers={"Origin": "https://example.invalid"})[0] == 403
        assert call("GET", "/v1/models", headers={"Host": "example.invalid"})[0] == 403
        assert call("POST", "/tokenize", {"content": "hi"}, headers={"Content-Type": "text/plain"})[0] == 415
        assert call("POST", "/tokenize", {}, headers={"Content-Length": str(MAX_REQUEST_BYTES + 1)})[0] == 413
        assert call("POST", "/tokenize", {"content": []})[0] == 400
        assert json.loads(call("POST", "/tokenize", {"content": "abc", "add_special": True})[2]) == {"tokens": [97, 98, 99, 999]}
        assert json.loads(call("POST", "/apply-template", {"messages": [{"content": "hello"}]})[2]) == {"prompt": "hello"}

        st, _, data = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        r = json.loads(data)
        msg = r["choices"][0]["message"]
        assert st == 200 and msg["content"] == "Hello world" and msg["reasoning_content"] == "thinking"
        assert r["choices"][0]["finish_reason"] == "stop" and r["usage"]["prompt_tokens_details"]["cached_tokens"] == 4

        st, ctype, data = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "read"}}],
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


def _call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        c.request(method, path, None if body is None else json.dumps(body),
                  {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        return r.status, r.getheader("Content-Type"), r.read()
    finally:
        c.close()


def test_network_auth_and_generation_admission():
    entered, release = threading.Event(), threading.Event()

    class BlockingEngine(FakeEngine):
        def run(self, job, emit, gone):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test timed out")
            return super().run(job, emit, gone)

    engine = BlockingEngine()
    try:
        make_handler(engine, "test", loopback=False)
    except ValueError:
        pass
    else:
        raise AssertionError("network listener must require a key")

    srv = BoundedHTTPServer(("127.0.0.1", 0), make_handler(engine, "test", api_key="test-key", loopback=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port
    auth = {"Authorization": "Bearer test-key"}
    job = {"messages": [{"role": "user", "content": "hello"}]}
    responses = []
    worker = threading.Thread(target=lambda: responses.append(_call(port, "POST", "/v1/chat/completions", job, auth)))
    try:
        assert _call(port, "GET", "/health")[0] == 401
        assert _call(port, "GET", "/health", headers=auth)[0] == 200
        assert _call(port, "POST", "/tokenize", {"content": "abc"})[0] == 401
        worker.start()
        assert entered.wait(5)
        assert _call(port, "POST", "/v1/chat/completions", job, auth)[0] == 503
        release.set()
        worker.join()
        assert responses[0][0] == 200
        assert _call(port, "POST", "/v1/chat/completions", job, auth)[0] == 200
    finally:
        release.set()
        if worker.ident is not None:
            worker.join()
        srv.shutdown()
        srv.server_close()


def test_ui_page_and_same_origin_requests():
    job = {"messages": [{"role": "user", "content": "hi"}]}

    srv = BoundedHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), "test"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port
    own = {"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin"}
    try:
        # Without --ui nothing changes: no page, and even this server's own origin is refused.
        assert _call(port, "GET", "/")[0] == 404
        assert _call(port, "POST", "/v1/chat/completions", job, own)[0] == 403
    finally:
        srv.shutdown()
        srv.server_close()

    srv = BoundedHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), "test", ui=True))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port
    own = {"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin"}
    try:
        st, ctype, page = _call(port, "GET", "/")
        assert st == 200 and ctype == "text/html; charset=utf-8" and b"/v1/chat/completions" in page
        assert _call(port, "GET", "/", headers={"Host": "example.invalid"})[0] == 403

        st, _, data = _call(port, "POST", "/v1/chat/completions", job, own)
        assert st == 200 and json.loads(data)["choices"][0]["message"]["content"] == "Hello world"
        assert _call(port, "GET", "/v1/models", headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
        ipv6 = {"Host": f"[::1]:{port}", "Origin": f"http://[::1]:{port}"}
        assert _call(port, "GET", "/v1/models", headers=ipv6)[0] == 200

        refused = [
            {"Origin": "https://example.invalid"},
            {"Origin": "null"},
            {"Origin": f"https://127.0.0.1:{port}"},
            {"Origin": f"http://localhost:{port}"},  # does not name the Host the request was sent to
            {"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "cross-site"},
            {"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-site"},
            # A rebinding name: Origin and Host agree, but the Host is not loopback.
            {"Origin": f"http://rebind.example.invalid:{port}", "Host": f"rebind.example.invalid:{port}"},
        ]
        for headers in refused:
            assert _call(port, "POST", "/v1/chat/completions", job, headers)[0] == 403, headers
    finally:
        srv.shutdown()
        srv.server_close()

    # On a network listener the page loads without the key, but its API calls still need it.
    srv = BoundedHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), "test", api_key="test-key", loopback=False, ui=True))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port
    own = {"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin"}
    try:
        assert _call(port, "GET", "/")[0] == 200
        assert _call(port, "POST", "/v1/chat/completions", job, own)[0] == 401
        assert _call(port, "POST", "/v1/chat/completions", job, {**own, "Authorization": "Bearer test-key"})[0] == 200
        assert _call(port, "GET", "/health", headers={"Origin": "https://example.invalid", "Authorization": "Bearer test-key"})[0] == 403

        # A TLS proxy preserves the browser's Host while forwarding HTTP to the backend.
        public = {"Host": "chat.example.test:8443", "Origin": "https://chat.example.test:8443",
                  "Sec-Fetch-Site": "same-origin"}
        auth = {"Authorization": "Bearer test-key"}
        assert _call(port, "GET", "/", headers={"Host": public["Host"]})[0] == 200
        assert _call(port, "POST", "/v1/chat/completions", job, public)[0] == 401
        assert _call(port, "POST", "/v1/chat/completions", job, {**public, **auth})[0] == 200
        default_tls = {"Host": "chat.example.test:443", "Origin": "https://chat.example.test"}
        assert _call(port, "GET", "/v1/models", headers={**default_tls, **auth})[0] == 200
        for origin in ("http://chat.example.test:8443", "https://chat.example.test", "https://other.example.test:8443"):
            assert _call(port, "POST", "/v1/chat/completions", job, {**public, **auth, "Origin": origin})[0] == 403
        assert _call(port, "GET", "/health", headers={"Host": public["Host"], **auth})[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_engine_keeps_a_model_thread_across_clients():
    class ThreadEngine(Engine):
        def _load(self, args):
            self.model_thread = threading.current_thread()

        def _run(self, job, emit, gone):
            return threading.current_thread()

    engine = ThreadEngine(None)
    threads = []
    try:
        for _ in range(2):
            client = threading.Thread(target=lambda: threads.append(engine.run(None, None, None)))
            client.start()
            client.join()
        assert threads == [engine.model_thread, engine.model_thread]
    finally:
        engine.close()
    assert not engine.model_thread.is_alive()


def main() -> None:
    test_output_parser()
    test_tool_calls_require_a_complete_declared_json_frame()
    test_prepare_carries_only_declared_tool_names()
    test_normalize_messages()
    test_http_round_trip()
    test_network_auth_and_generation_admission()
    test_ui_page_and_same_origin_requests()
    test_engine_keeps_a_model_thread_across_clients()
    print("ALL LAMQ SERVE TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
