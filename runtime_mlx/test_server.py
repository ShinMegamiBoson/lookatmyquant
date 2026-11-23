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
    MAX_REQUEST_BYTES, SPECIAL, BoundedHTTPServer, Engine, OutputParser, RequestError,
    anthropic_to_openai, apply_repetition_guard, make_handler, normalize_messages, responses_to_openai,
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
        assert _call(port, "POST", "/v1/chat/completions", {**job, "model": "test"}, own)[0] == 200
        old_chat = {**job, "model": "previous-model"}
        st, _, data = _call(port, "POST", "/v1/chat/completions", old_chat, own)
        assert st == 409 and json.loads(data)["error"]["type"] == "model_changed"
        assert _call(port, "POST", "/v1/chat/completions", old_chat)[0] == 200
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


def test_anthropic_messages_endpoint() -> None:
    srv = BoundedHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), "inkling-test", api_key="secret", loopback=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port
    hdr = {"x-api-key": "secret"}
    try:
        # Translation check: system blocks, tool_use, mixed tool_result + text user turn
        oai = anthropic_to_openai({
            "system": [{"type": "text", "text": "Be terse."}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "tool_use", "id": "tu_1", "name": "read", "input": {"path": "a.py"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "print(1)"},
                    {"type": "text", "text": "now explain"},
                ]},
            ],
            "tools": [{"name": "read", "description": "read file", "input_schema": {"type": "object"}}],
            "thinking": {"type": "disabled"},
            "max_tokens": 64,
        })
        assert oai["messages"][0] == {"role": "system", "content": "Be terse."}
        assert oai["messages"][2]["reasoning_content"] == "plan"
        assert oai["messages"][3] == {"role": "tool", "tool_call_id": "tu_1", "content": "print(1)"}
        assert oai["messages"][4] == {"role": "user", "content": "now explain"}
        assert oai["reasoning_effort"] == "none"
        assert oai["tools"][0]["function"]["name"] == "read"

        # Auth check: rejected without key, accepted with x-api-key
        assert _call(port, "POST", "/v1/messages", {"messages": [{"role": "user", "content": "hi"}]})[0] == 401
        st, _, data = _call(port, "POST", "/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}, hdr)
        assert st == 200
        body = json.loads(data)
        assert body["type"] == "message" and body["stop_reason"] == "end_turn"
        assert body["content"] == [{"type": "thinking", "thinking": "thinking"}, {"type": "text", "text": "Hello world"}]
        assert body["usage"] == {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 4}

        # Streaming with tool_use
        st, ctype, data = _call(port, "POST", "/v1/messages", {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            "stream": True,
        }, hdr)
        assert st == 200 and ctype == "text/event-stream"
        ev_names = [ln[7:] for ln in data.decode().splitlines() if ln.startswith("event: ")]
        assert ev_names[0] == "message_start" and ev_names[-1] == "message_stop"
        assert "content_block_start" in ev_names and "message_delta" in ev_names
        assert '"tool_use"' in data.decode() and '"input_json_delta"' in data.decode()
    finally:
        srv.shutdown()
        srv.server_close()


def test_openai_responses_endpoint() -> None:
    srv = BoundedHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), "inkling-test"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port
    try:
        oai = responses_to_openai({
            "instructions": "Use tools carefully.",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect"}]},
                {"type": "reasoning", "summary": [{"text": "checking"}]},
                {"type": "function_call", "call_id": "fc_1", "name": "read", "arguments": '{"path":"x"}'},
                {"type": "function_call_output", "call_id": "fc_1", "output": "ok"},
            ],
            "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}],
            "reasoning": {"effort": "low"},
            "max_output_tokens": 128,
        })
        assert oai["messages"][0] == {"role": "system", "content": "Use tools carefully."}
        assert oai["messages"][2]["tool_calls"][0]["function"]["name"] == "read"
        assert oai["messages"][2]["reasoning_content"] == "checking"
        assert oai["messages"][3] == {"role": "tool", "tool_call_id": "fc_1", "content": "ok"}
        assert oai["reasoning_effort"] == "low" and oai["max_tokens"] == 128

        st, _, data = _call(port, "POST", "/v1/responses", {"input": "hi"})
        assert st == 200
        body = json.loads(data)
        assert body["object"] == "response" and body["status"] == "completed" and body["output_text"] == "Hello world"
        assert [it["type"] for it in body["output"]] == ["reasoning", "message"]
        assert body["usage"]["input_tokens_details"]["cached_tokens"] == 4

        st, ctype, data = _call(port, "POST", "/v1/responses", {
            "input": "hi",
            "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}],
            "stream": True,
        })
        assert st == 200 and ctype == "text/event-stream"
        txt = data.decode()
        assert "event: response.created" in txt and "event: response.completed" in txt
        assert "event: response.function_call_arguments.delta" in txt
    finally:
        srv.shutdown()
        srv.server_close()


def test_repetition_guard() -> None:
    def make_loop(n, *, result="ok", vary_cmd=False):
        msgs = [{"role": "user", "content": "fix bug"}]
        for i in range(n):
            cmd = f"step_{i}_unique_{chr(97 + i)}" if vary_cmd else f"echo end{i}"
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": {"cmd": cmd}}}],
            })
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": result})
        return msgs

    # 5 calls -> no trigger
    m5 = make_loop(5)
    out5, info5 = apply_repetition_guard(m5)
    assert info5 is None and out5 is m5

    # 6 near-identical calls (digits ignored) + short output -> triggers without mutating input
    m6 = make_loop(6)
    out6, info6 = apply_repetition_guard(m6, endpoint="messages")
    assert info6 == {"tool": "bash", "streak": 6, "endpoint": "messages"}
    assert len(m6) == 13 and len(out6) == 14
    assert out6[-1]["role"] == "user" and "bash" in out6[-1]["content"] and "6 times" in out6[-1]["content"]

    # Long tool result (> 200 chars) -> no trigger
    m_long = make_loop(6, result="x" * 201)
    assert apply_repetition_guard(m_long)[1] is None

    # Distinct non-digit args -> no trigger
    m_var = make_loop(6, vary_cmd=True)
    assert apply_repetition_guard(m_var)[1] is None


def test_disk_prompt_cache_delta_round_trip() -> None:
    import tempfile
    import mlx.core as mx
    from runtime_mlx.inkling.model import Inkling2Cache, Inkling2Config
    from runtime_mlx.prompt_cache_disk import DiskPromptCache

    class TinyModel:
        def __init__(self, tmp: Path):
            self.packed_dir = tmp / "model"
            self.packed_dir.mkdir(exist_ok=True)
            (self.packed_dir / "manifest.json").write_text('{"v":1}')
            self.cfg = Inkling2Config(
                hidden_size=16, num_hidden_layers=2, vocab_size=64, unpadded_vocab_size=64,
                num_attention_heads=2, num_key_value_heads=2, head_dim=8, d_rel=4, rel_extent=8,
                local_layer_ids=(0,), sliding_window_size=4, n_layers_per_attn_res_block=2,
                n_routed_experts=2, num_experts_per_tok=1, intermediate_size=8,
                loop_start=1, loop_end=1, loop_passes=1,
            )

        def make_cache(self):
            return Inkling2Cache(self.cfg)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model = TinyModel(root)
        slot_dir = root / "slots"
        dc = DiskPromptCache(slot_dir, max_entries=1, min_tokens=4, max_disk_gb=1.0)
        dc._owner = (model, None)

        # Save base snapshot at 6 tokens
        c1 = model.make_cache()
        c1.seq_len = 6
        c1.kv_is_sliding = [True, False]
        c1.keys = [mx.arange(2 * 4 * 8, dtype=mx.float32).reshape(1, 2, 4, 8),
                   mx.arange(2 * 6 * 8, dtype=mx.float32).reshape(1, 2, 6, 8)]
        c1.values = [c1.keys[0] + 1.0, c1.keys[1] + 2.0]
        c1.conv_states[0][0] = mx.ones((1, 3, 16), dtype=mx.float32)
        dc._store([1, 2, 3, 4, 5, 6], c1, None)

        # Save delta extension at 9 tokens
        c2 = model.make_cache()
        c2.seq_len = 9
        c2.kv_is_sliding = [True, False]
        k_tail = mx.arange(2 * 3 * 8, dtype=mx.float32).reshape(1, 2, 3, 8) + 100.0
        v_tail = k_tail + 5.0
        c2.keys = [c1.keys[0] + 10.0, mx.concatenate([c1.keys[1], k_tail], axis=2)]
        c2.values = [c1.values[0] + 10.0, mx.concatenate([c1.values[1], v_tail], axis=2)]
        c2.conv_states[0][0] = mx.ones((1, 3, 16), dtype=mx.float32) * 3.0
        dc._store([1, 2, 3, 4, 5, 6, 7, 8, 9], c2, None)

        metas = sorted(slot_dir.glob("*.meta.json"))
        assert len(metas) == 2
        m_list = [json.loads(p.read_text()) for p in metas]
        assert any(m["parent_id"] is not None and m["delta_starts"] == [0, 6] for m in m_list)

        # Simulate server restart with a brand new DiskPromptCache
        dc2 = DiskPromptCache(slot_dir, max_entries=2, min_tokens=4, max_disk_gb=1.0)
        dc2._owner = (model, None)
        hit = dc2._match([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], drafting=False)
        assert hit is not None and hit["tokens"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        rc = hit["cache"]
        assert rc.seq_len == 9
        assert float(mx.abs(rc.keys[1] - c2.keys[1]).max()) == 0.0
        assert float(mx.abs(rc.values[1] - c2.values[1]).max()) == 0.0
        assert float(mx.abs(rc.keys[0] - c2.keys[0]).max()) == 0.0
        assert float(mx.abs(rc.conv_states[0][0] - c2.conv_states[0][0]).max()) == 0.0


def main() -> None:
    test_output_parser()
    test_tool_calls_require_a_complete_declared_json_frame()
    test_prepare_carries_only_declared_tool_names()
    test_normalize_messages()
    test_http_round_trip()
    test_network_auth_and_generation_admission()
    test_ui_page_and_same_origin_requests()
    test_engine_keeps_a_model_thread_across_clients()
    test_anthropic_messages_endpoint()
    test_openai_responses_endpoint()
    test_repetition_guard()
    test_disk_prompt_cache_delta_round_trip()
    print("ALL LAMQ SERVE TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
