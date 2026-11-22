"""OpenAI-compatible HTTP server for packed Inkling-2-Mini models (`lamq serve`).

    lamq serve shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB \\
        --drafter shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter

Endpoints: `POST /v1/chat/completions` (streaming and not), `GET /v1/models`, `GET /health`.

* Requests are rendered with the model's own chat template, including `tools` and
  `reasoning_effort` (none, minimal, low, medium, high, xhigh, max).
* The model's output is parsed back into OpenAI fields: thinking -> `reasoning_content`, text ->
  `content`, `<|content_invoke_tool_json|>` messages -> `tool_calls` (`finish_reason: tool_calls`).
* A `PromptCache` keeps prefilled KV snapshots, so a request that extends an earlier prompt (every
  turn of a chat or agent loop) prefills only its new tokens; `usage.prompt_tokens_details.cached_tokens`
  reports the reuse.
* Prompts longer than `--context` are refused with llama.cpp's error wording, which agent clients
  (for example Pi) already treat as "compact the conversation and retry".

One request is generated at a time; concurrent requests wait their turn.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
SPECIAL = {
    "user": "<|message_user|>",
    "model": "<|message_model|>",
    "system": "<|message_system|>",
    "tool": "<|message_tool|>",
    "text": "<|content_text|>",
    "thinking": "<|content_thinking|>",
    "eos": "<|content_model_end_sampling|>",
    "end": "<|end_message|>",
    "tool_json": "<|content_invoke_tool_json|>",
    "tool_text": "<|content_invoke_tool_text|>",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class ClientGone(Exception):
    """The HTTP client disconnected."""


class RequestError(Exception):
    def __init__(self, status: int, message: str, etype: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.etype = status, message, etype


# --------------------------------------------------------------------------- request -> template


def normalize_messages(msgs: Any) -> list[dict]:
    """Convert OpenAI chat messages into the shape Inkling's chat template expects.

    The template wants tool-call arguments as objects (OpenAI sends JSON strings), plain-string
    content, and `reasoning_content` on assistant turns to replay earlier thinking.
    """
    if not isinstance(msgs, list) or not msgs:
        raise RequestError(400, "messages must be a non-empty list")
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            raise RequestError(400, "each message must be an object")
        role = "system" if m.get("role") == "developer" else m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise RequestError(400, f"unsupported role: {m.get('role')!r}")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and p.get("type") in (None, "text", "input_text"):
                    parts.append(p.get("text") or "")
                elif isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"):
                    parts.append("[image omitted: this server is text-only]")
            content = "".join(parts)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        nm: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            rc = m.get("reasoning_content") or m.get("reasoning") or m.get("reasoning_text")
            if isinstance(rc, str) and rc.strip():
                nm["reasoning_content"] = rc
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"input": args}
                if not isinstance(args, dict):
                    args = {"input": args}
                calls.append({"id": tc.get("id"), "type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
            if calls:
                nm["tool_calls"] = calls
        elif role == "tool":
            if m.get("tool_call_id"):
                nm["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                nm["name"] = m["name"]
        out.append(nm)
    return out


# --------------------------------------------------------------------------- output parsing


class OutputParser:
    """Token-level parser for Inkling's output grammar.

        <|message_model|><|content_thinking|>...<|end_message|>                -> reasoning
        <|message_model|><|content_text|>...<|end_message|>                    -> content
        <|message_model|>NAME<|content_invoke_tool_json|>{json}<|end_message|> -> one tool call
        <|content_model_end_sampling|>                                         -> end of turn

    Parsing stops after the first tool call: left to continue, the model tends to start a fresh
    turn and repeat the call. `feed()` returns events `("reasoning"|"content", text)` and
    `("tool", openai_tool_call)`.
    """

    def __init__(self, tokenizer, ids: dict[str, int]):
        self.tok = tokenizer
        self.I = ids
        self.other_roles = {ids["user"], ids["system"], ids["tool"]}
        self.state = "between"
        self.header: list[int] = []
        self.pending: list[int] = []
        self.tool_tokens: list[int] = []
        self.tool_name = ""
        self.new_segment = False
        self.seen = {"reasoning": False, "content": False}
        self.done = False
        self.finish = "stop"

    def _dec(self, toks: list[int]) -> str:
        return self.tok.decode(toks, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _enter(self, state: str) -> None:
        self.state, self.new_segment = state, True

    def _emit(self, s: str, out: list) -> None:
        kind = "reasoning" if self.state == "thinking" else "content"
        if self.new_segment and self.seen[kind]:
            s = "\n\n" + s  # consecutive thinking (or text) messages
        self.new_segment = False
        self.seen[kind] = True
        out.append((kind, s))

    def _push(self, t: int, out: list) -> None:
        self.pending.append(t)
        s = self._dec(self.pending)
        if s and not s.endswith("�"):  # hold back incomplete UTF-8
            self.pending = []
            self._emit(s, out)

    def _flush(self, out: list) -> None:
        if self.pending:
            s = self._dec(self.pending).replace("�", "")
            self.pending = []
            if s:
                self._emit(s, out)

    def _finish_tool(self, out: list) -> None:
        raw = self._dec(self.tool_tokens).strip()
        name, args = self.tool_name, None
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                name = obj.get("name") or name
                args = obj["args"] if "args" in obj else {k: v for k, v in obj.items() if k != "name"}
        except json.JSONDecodeError:
            pass
        arguments = json.dumps(args, ensure_ascii=False) if args is not None else raw
        out.append(("tool", {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                             "function": {"name": name, "arguments": arguments}}))
        self.done, self.finish = True, "tool_calls"

    def feed(self, t: int) -> list:
        I, out, st = self.I, [], self.state
        if self.done:
            return out
        if t == I["eos"] or t in self.other_roles:
            if st in ("thinking", "text"):
                self._flush(out)
            elif st == "tool" and t == I["eos"]:
                self._finish_tool(out)
                return out
            self.done = True
            return out
        if st == "between":
            if t == I["model"]:
                self.state, self.header = "header", []
            elif t in (I["thinking"], I["text"]):
                self._enter("thinking" if t == I["thinking"] else "text")
            elif t != I["end"]:
                self._enter("text")
                self._push(t, out)
        elif st == "header":
            if t in (I["thinking"], I["text"]):
                self._enter("thinking" if t == I["thinking"] else "text")
            elif t in (I["tool_json"], I["tool_text"]):
                self.state, self.tool_tokens = "tool", []
                self.tool_name = self._dec(self.header).strip()
            elif t == I["end"]:
                self.state = "between"
            else:
                self.header.append(t)
                if len(self.header) > 64:  # not a header after all
                    self._enter("text")
                    for h in self.header:
                        self._push(h, out)
        elif st in ("thinking", "text"):
            if t == I["end"]:
                self._flush(out)
                self.state = "between"
            elif t == I["model"]:
                self._flush(out)
                self.state, self.header = "header", []
            elif t in (I["thinking"], I["text"]):
                self._flush(out)
                self._enter("thinking" if t == I["thinking"] else "text")
            else:
                self._push(t, out)
        elif st == "tool":
            if t == I["end"]:
                self._finish_tool(out)
            else:
                self.tool_tokens.append(t)
        return out

    def close(self) -> list:
        out: list = []
        if self.state in ("thinking", "text"):
            self._flush(out)
        return out


# --------------------------------------------------------------------------- engine


class Engine:
    """Loads the model once and generates for one request at a time."""

    def __init__(self, args: argparse.Namespace):
        import mlx.core as mx

        from .inkling import PromptCache
        from .inkling.generate import _unwrap_model
        from .runtime import Runtime, detect_format, resolve_path

        self.args = args
        path = resolve_path(args.model)
        fmt = detect_format(path)
        if fmt != "inkling2-packed":
            raise SystemExit(f"lamq serve currently supports packed Inkling-2-Mini models (inkling2-packed); {args.model} is {fmt}")
        t = time.perf_counter()
        self.rt = Runtime.load(path, drafter=args.drafter, verify=not args.no_verify)
        self.tok = self.rt.tokenizer
        self.hf = getattr(self.tok, "_tokenizer", self.tok)
        self.m = _unwrap_model(self.rt.model)
        self.drafter = self.rt.drafter
        self.ids = {}
        for k, s in SPECIAL.items():
            i = self.hf.convert_tokens_to_ids(s)
            if not isinstance(i, int) or i < 0:
                raise SystemExit(f"tokenizer has no {s}; is this an Inkling-2 model?")
            self.ids[k] = i
        self.prompt_cache = PromptCache(max_entries=args.prompt_cache)
        self.lock = threading.Lock()
        log(f"loaded {args.model} ({'with drafter' if self.drafter is not None else 'serial'}) in "
            f"{time.perf_counter() - t:.1f}s, {mx.get_active_memory() / 1e9:.2f} GB")

    def prepare(self, req: dict) -> dict:
        """Validate and tokenize a request (raises RequestError before any response bytes are sent)."""
        messages = normalize_messages(req.get("messages"))
        tools = req.get("tools") or None
        effort = req.get("reasoning_effort")
        if isinstance(effort, dict):
            effort = effort.get("effort")
        if isinstance(effort, str) and effort.strip().lower() in EFFORTS:
            effort = effort.strip().lower()
        elif not isinstance(effort, (int, float)) or isinstance(effort, bool):
            effort = self.args.reasoning_effort
        kw: dict[str, Any] = dict(tokenize=True, add_generation_prompt=True, reasoning_effort=effort)
        if tools:
            kw["tools"] = tools
        try:
            enc = self.tok.apply_chat_template(messages, **kw)
        except Exception as e:  # jinja raise_exception() and friends
            raise RequestError(400, f"chat template error: {e}")
        if isinstance(enc, dict) or hasattr(enc, "input_ids"):
            enc = enc["input_ids"]
        ids = [int(x) for x in enc]
        ctx = self.args.context
        if len(ids) >= ctx:
            raise RequestError(400, f"the request exceeds the available context size, try increasing it "
                                    f"(prompt {len(ids)} tokens, context {ctx})", "exceed_context_size_error")
        max_tokens = req.get("max_completion_tokens") or req.get("max_tokens") or self.args.max_tokens
        temp = req.get("temperature")
        top_p = req.get("top_p")
        return {
            "ids": ids,
            "effort": effort,
            "max_tokens": max(1, min(int(max_tokens), ctx - len(ids))),
            "temp": self.args.temp if temp is None else float(temp),
            "top_p": self.args.top_p if top_p is None else float(top_p),
            "top_k": int(req.get("top_k") or 0),
            "use_cache": req.get("cache_prompt", True) is not False,
        }

    def run(self, job: dict, emit: Callable[[str, Any], None], gone: Callable[[], bool]) -> tuple[dict, str]:
        """Generate for a prepared request, sending parsed events to `emit`. Returns `(usage, finish_reason)`."""
        import mlx.core as mx

        from .inkling import stream_dspark, stream_serial

        ids = job["ids"]
        with self.lock:
            stats: dict[str, Any] = {}
            common = dict(max_tokens=job["max_tokens"], eos_ids={self.ids["eos"]}, temp=job["temp"], top_p=job["top_p"],
                          top_k=job["top_k"], stats_out=stats, prompt_cache=self.prompt_cache if job["use_cache"] else None)
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            if self.drafter is not None:
                gen = stream_dspark(self.m, self.drafter, ids, **common)
            else:
                gen = stream_serial(self.m, ids, **common)
            parser = OutputParser(self.hf, self.ids)
            n_gen, finish, last_check = 0, None, time.perf_counter()
            try:
                for chunk in gen:
                    for t in chunk:
                        n_gen += 1
                        for ev in parser.feed(int(t)):
                            emit(*ev)
                        if parser.done or n_gen >= job["max_tokens"]:
                            break
                    if parser.done or n_gen >= job["max_tokens"]:
                        break
                    if time.perf_counter() - last_check > 2.0:
                        last_check = time.perf_counter()
                        if gone():
                            raise ClientGone()
                for ev in parser.close():
                    emit(*ev)
                finish = parser.finish if parser.done else "length"
            finally:
                gen.close()
                elapsed = time.perf_counter() - t0
                reused = int(stats.get("prompt_cached_tokens", 0) or 0)
                prompt_s = float(stats.get("prompt_s", 0.0) or 0.0)
                log(f"prompt {len(ids)} tok (cached {reused}, prefill {prompt_s:.1f}s) | gen {n_gen} tok "
                    f"({n_gen / max(elapsed - prompt_s, 1e-9):.1f} tok/s) | effort {job['effort']} temp {job['temp']} | "
                    f"{finish or 'aborted'} | peak {mx.get_peak_memory() / 1e9:.2f} GB")
                mx.clear_cache()
        usage = {"prompt_tokens": len(ids), "completion_tokens": n_gen, "total_tokens": len(ids) + n_gen,
                 "prompt_tokens_details": {"cached_tokens": reused}}
        return usage, finish

    def warmup(self) -> None:
        """Compile the Metal kernels for greedy and sampled paths before the first real request."""
        t = time.perf_counter()
        for temp in (0.0, 0.7):
            job = self.prepare({"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 4, "temperature": temp})
            job["use_cache"] = False
            self.run(job, lambda *a: None, lambda: False)
        log(f"warmup {time.perf_counter() - t:.1f}s")


# --------------------------------------------------------------------------- HTTP


def make_handler(engine, model_id: str):
    """Request handler bound to an engine with `prepare`, `run` and `args.context`."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "lamq-serve"

        def log_message(self, fmt, *a):
            pass

        def _json(self, status: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, etype: str = "invalid_request_error") -> None:
            self._json(status, {"error": {"message": message, "type": etype, "code": status}})

        def _gone(self) -> bool:
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in (("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Headers", "*"),
                         ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"), ("Content-Length", "0")):
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model", "created": int(time.time()),
                                                             "owned_by": "lamq", "context_length": engine.args.context}]})
            else:
                self._error(404, f"not found: {self.path}")

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._error(404, f"not found: {self.path}")
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("request body must be a JSON object")
            except (ValueError, json.JSONDecodeError) as e:
                self._error(400, f"bad JSON: {e}")
                return
            try:
                job = engine.prepare(req)
            except RequestError as e:
                self._error(e.status, e.message, e.etype)
                return
            except Exception as e:
                traceback.print_exc()
                self._error(500, f"server error: {e}", "server_error")
                return
            (self._stream if req.get("stream") else self._complete)(req, job)

        def _complete(self, req: dict, job: dict) -> None:
            text = {"reasoning": [], "content": []}
            calls: list = []

            def emit(kind, payload):
                (calls if kind == "tool" else text[kind]).append(payload)

            try:
                usage, finish = engine.run(job, emit, self._gone)
            except ClientGone:
                return
            except Exception as e:
                traceback.print_exc()
                self._error(500, f"server error: {e}", "server_error")
                return
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(text["content"]) or (None if calls else "")}
            if text["reasoning"]:
                msg["reasoning_content"] = "".join(text["reasoning"])
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                             "model": model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage})

        def _stream(self, req: dict, job: dict) -> None:
            cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            wlock, started, broken = threading.Lock(), threading.Event(), threading.Event()
            n_tools = [0]

            def write(raw: bytes) -> None:
                with wlock:
                    try:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(raw), raw))
                        self.wfile.flush()
                    except OSError:
                        broken.set()
                        raise ClientGone()

            def chunk(delta, finish=None, usage=None) -> None:
                obj: dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

            def emit(kind, payload) -> None:
                started.set()
                if kind == "tool":
                    chunk({"tool_calls": [dict(payload, index=n_tools[0])]})
                    n_tools[0] += 1
                else:
                    chunk({"reasoning_content" if kind == "reasoning" else "content": payload})

            def keepalive() -> None:  # SSE comments while a long prompt prefills, so clients don't time out
                while not started.wait(10.0):
                    try:
                        write(b": prefilling\n\n")
                    except ClientGone:
                        return

            self.send_response(200)
            for k, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
                         ("Transfer-Encoding", "chunked"), ("Access-Control-Allow-Origin", "*")):
                self.send_header(k, v)
            self.end_headers()
            ka = threading.Thread(target=keepalive, daemon=True)
            try:
                chunk({"role": "assistant", "content": ""})
                ka.start()
                usage, finish = engine.run(job, emit, lambda: broken.is_set() or self._gone())
                started.set()
                chunk({}, finish)
                if include_usage:
                    chunk(None, usage=usage)
                write(b"data: [DONE]\n\n")
                write(b"")
            except ClientGone:
                log("client disconnected; generation stopped")
                self.close_connection = True
            except Exception as e:
                traceback.print_exc()
                try:
                    write(b"data: " + json.dumps({"error": {"message": f"server error: {e}", "type": "server_error",
                                                            "code": 500}}).encode() + b"\n\n")
                    write(b"data: [DONE]\n\n")
                    write(b"")
                except ClientGone:
                    pass
            finally:
                started.set()

    return Handler


def add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="packed Inkling-2-Mini directory or Hugging Face repo ID")
    p.add_argument("--drafter", default=None, help="DSpark drafter directory, .safetensors file or Hugging Face repo ID")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; 0.0.0.0 for the LAN, no auth)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model-id", default=None, help="model id reported by /v1/models (default: the model's name)")
    p.add_argument("--context", type=int, default=32768, help="prompt + completion token limit (default 32768)")
    p.add_argument("--max-tokens", type=int, default=8192, help="completion tokens when a request sets none")
    p.add_argument("--reasoning-effort", default="medium", choices=EFFORTS, help="when a request sets none")
    p.add_argument("--temp", type=float, default=0.0, help="temperature when a request sets none (0 = greedy)")
    p.add_argument("--top-p", type=float, default=1.0, help="top-p when a request sets none")
    p.add_argument("--prompt-cache", type=int, default=2, help="prompt-prefix KV snapshots kept (0 disables)")
    p.add_argument("--no-verify", action="store_true", help="skip SHA-256 checksum verification when loading")
    p.add_argument("--no-warmup", action="store_true")


def serve(args: argparse.Namespace) -> int:
    engine = Engine(args)
    if not args.no_warmup:
        engine.warmup()
    model_id = args.model_id or str(args.model).rstrip("/").split("/")[-1]
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine, model_id))
    srv.daemon_threads = True
    log(f"serving {model_id!r} on http://{args.host}:{args.port}/v1 (context {args.context}, "
        f"reasoning_effort {args.reasoning_effort}, temp {args.temp})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lamq serve", description="OpenAI-compatible server for packed Inkling-2-Mini models.")
    add_serve_args(p)
    return serve(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
