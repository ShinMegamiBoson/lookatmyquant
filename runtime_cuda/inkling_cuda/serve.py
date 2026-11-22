"""Loopback OpenAI-compatible chat and tokenizer endpoints for Inkling CUDA."""
from contextlib import closing
from dataclasses import dataclass
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from .device_model import DeviceModel
from .driver import Cuda


END = "<|end_message|>"
MODEL = "<|message_model|>"
TOOL = "<|content_invoke_tool_json|>"
TEXT = "<|content_text|>"
THINK = "<|content_thinking|>"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
LONG_PROMPT_TOKENS = 512
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def render_chat(tokenizer, request, *, no_thinking, effort=None):
    if effort is None:
        effort = request.get("reasoning_effort", "none")
    messages = []
    for original in request["messages"]:
        message = dict(original)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                arguments = function.get("arguments", {})
                function["arguments"] = json.loads(arguments) if isinstance(arguments, str) else arguments
                calls.append({**call, "function": function})
            message["tool_calls"] = calls
        messages.append(message)
    return tokenizer.apply_chat_template(
        messages, tools=request.get("tools") or None,
        add_generation_prompt=request.get("add_generation_prompt", True),
        reasoning_effort="none" if no_thinking else effort,
        tokenize=False,
    )


def assistant_message(text, tools, completion_id):
    if tools and TOOL in text:
        prefix, body = text.rsplit(TOOL, 1)
        raw, delimiter, _ = body.partition(END)
        if delimiter:
            raw = raw.strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as error:
                payload = None
                # Some completed Inkling frames omit just the outer closing brace.
                if error.pos == len(raw):
                    try:
                        payload = json.loads(raw + "}")
                    except json.JSONDecodeError:
                        pass
            if isinstance(payload, dict):
                name = payload.get("name")
                if isinstance(name, str):
                    arguments = payload.get("args", payload.get("arguments"))
                else:
                    name = prefix.rsplit(MODEL, 1)[-1].strip()
                    arguments = payload.get("arguments", payload)
                allowed = {tool["function"]["name"] for tool in tools}
                if name in allowed and isinstance(arguments, dict):
                    call = {"id": f"call_{completion_id}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
                    return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
    if TEXT in text and TOOL not in text:
        text = text.rsplit(TEXT, 1)[-1].split(END, 1)[0]
    return {"role": "assistant", "content": text}, "stop"


class Frames:
    """Translate Inkling's token-delimited channels into OpenAI message deltas."""

    def __init__(self, chat, tools, completion_id):
        from tokenizers.decoders import DecodeStream

        self.stream_type = DecodeStream
        self.chat, self.tools, self.completion_id = chat, tools, completion_id
        self.tool_name = ""
        self.reset("content")

    def reset(self, channel):
        self.channel = channel
        self.pending = []
        self.decoder = self.stream_type(skip_special_tokens=False)

    def decode_pending(self):
        return self.chat.tokenizer.decode(self.pending, skip_special_tokens=False) if self.pending else ""

    def finish(self):
        if self.channel == "header" and self.pending:
            return {"content": self.decode_pending()}
        return {}

    def token(self, token):
        chat = self.chat
        if token == chat.end:
            delta = self.finish()
            terminal = self.channel in ("content", "header")
            if self.channel == "tool":
                # Only completed, declared calls are emitted as executable tool deltas.
                message, _ = assistant_message(MODEL + self.tool_name + TOOL + self.decode_pending() + END,
                                               self.tools, self.completion_id)
                if message.get("tool_calls"):
                    delta = {"tool_calls": [dict(index=0, **message["tool_calls"][0])]}
                return delta, "tool_calls" if delta else "stop"
            self.reset("content")
            return delta, "stop" if terminal else None
        if token == chat.tool:
            self.tool_name = self.decode_pending()
            self.reset("tool")
        elif token == chat.message:
            self.reset("header")
        elif token == chat.text:
            self.reset("content")
        elif token == chat.thinking:
            self.reset("reasoning_content")
        elif self.channel in ("header", "tool"):
            self.pending.append(token)
        else:
            piece = self.decoder.step(chat.tokenizer.backend_tokenizer, token)
            if piece:
                return {self.channel: piece}, None
        return {}, None


@dataclass
class Completion:
    prompt: list
    limit: int
    tools: list
    stop_at_text_end: bool
    blocked_token: int | None
    temperature: float
    seed: int | None


class Chat:
    def __init__(self, tokenizer, model, eos, context, name, no_thinking, *, drafter=None, draft_block_size=2, device=0):
        self.tokenizer = tokenizer
        self.model = model
        self.eos = eos
        self.context = context
        self.name = name
        self.no_thinking = no_thinking
        self.end, self.tool, self.thinking, self.message, self.text = tokenizer.convert_tokens_to_ids(
            [END, TOOL, THINK, MODEL, TEXT],
        )
        self.drafter, self.draft_block_size, self.device = drafter, draft_block_size, device
        self.calls = 0

    def render(self, request):
        return self.prompt(request)[0]

    def tokenize(self, text, add_special=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special)

    def prompt(self, request):
        template = request.get("chat_template_kwargs") or {}
        if not isinstance(template, dict):
            raise ValueError("chat_template_kwargs must be an object")
        effort = request.get("reasoning_effort", template.get("reasoning_effort"))
        automatic = effort is None and "enable_thinking" not in template
        if self.no_thinking or template.get("enable_thinking") is False:
            effort = "none"
        elif effort is None:
            effort = "low"
        if isinstance(effort, str):
            effort = effort.strip()
            if effort not in EFFORTS:
                raise ValueError("Unknown reasoning effort")
        elif type(effort) not in (float, int) or not 0 <= effort <= 0.99:
            raise ValueError("Reasoning effort must be in [0, 0.99]")
        rendered = render_chat(self.tokenizer, request, no_thinking=False, effort=effort)
        ids = self.tokenize(rendered)
        # With automatic effort, prefer a direct answer on long documents.
        # Callers can still explicitly ask for a reasoning trace.
        if automatic and effort != "none" and not request.get("tools") and len(ids) >= LONG_PROMPT_TOKENS:
            effort = "none"
            rendered = render_chat(self.tokenizer, request, no_thinking=False, effort=effort)
            ids = self.tokenize(rendered)
        return rendered, ids, self.thinking if effort in ("none", 0) else None

    def prepare(self, request):
        if request.get("model", self.name) != self.name:
            raise ValueError("Unknown model")
        _, ids, blocked = self.prompt(request)
        available = self.context - len(ids) + 1
        limit = request.get("max_completion_tokens", request.get("max_tokens"))
        if limit is None:
            limit = available
        if type(limit) is not int or limit < 1:
            raise ValueError("Completion limit must be a positive integer")
        if not ids or limit > available:
            raise ValueError("Prompt and completion allowance exceed the configured context")
        temperature = request.get("temperature", 0)
        if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
            raise ValueError("Temperature must be in [0, 2]")
        seed = request.get("seed")
        if seed is not None and (type(seed) is not int or not -(1 << 63) <= seed < (1 << 64)):
            raise ValueError("Seed must be a 64-bit integer")
        tools = request.get("tools") or []
        messages = request["messages"]
        return Completion(ids, limit, tools, not tools or bool(messages and messages[-1]["role"] == "tool"),
                          blocked, temperature, seed)

    def tokens(self, completion):
        context = len(completion.prompt) + completion.limit - 1
        with self.model.session(context) as session:
            sampler = None
            if completion.temperature:
                from .drafter import TemperatureSampler

                sampler = TemperatureSampler(session, f"cuda:{self.device}", completion.temperature, completion.seed)
            if self.drafter is not None:
                from .drafter import stream_speculative

                yield from stream_speculative(
                    self.drafter, session, completion.prompt, completion.limit, self.eos, self.draft_block_size,
                    blocked_token=completion.blocked_token, sampler=sampler,
                )
                return
            session.prefill(completion.prompt[:-1])
            token = completion.prompt[-1]
            for _ in range(completion.limit):
                token = session.predict(token, blocked_token=completion.blocked_token, sampler=sampler)
                yield token
                if token in self.eos:
                    return

    def chunks(self, completion):
        completion_id = f"inkling_{self.calls}"
        self.calls += 1
        frames = Frames(self, completion.tools, completion_id)
        base = {"id": f"chatcmpl-{completion_id}", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": self.name}

        def chunk(delta, finish=None):
            return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        yield chunk({"role": "assistant"})
        finish, count = "length", 0
        with closing(self.tokens(completion)) as tokens:
            for count, token in enumerate(tokens, 1):
                if token in self.eos:
                    finish = "stop"
                    break
                delta, terminal = frames.token(token)
                if delta:
                    yield chunk(delta)
                if terminal is not None and (completion.stop_at_text_end or terminal == "tool_calls"
                                             or frames.channel == "tool"):
                    finish = terminal
                    break
        if tail := frames.finish():
            yield chunk(tail)
        yield chunk({}, finish)
        yield {**base, "choices": [],
               "usage": {"prompt_tokens": len(completion.prompt), "completion_tokens": count,
                         "total_tokens": len(completion.prompt) + count}}

    def complete(self, request):
        message = {"role": "assistant", "content": None}
        for chunk in self.chunks(self.prepare(request)):
            if not chunk["choices"]:
                usage = chunk["usage"]
                break
            choice, = chunk["choices"]
            for key, value in choice["delta"].items():
                if key in ("content", "reasoning_content"):
                    message[key] = (message.get(key) or "") + value
                elif key == "tool_calls":
                    message[key] = [{k: v for k, v in call.items() if k != "index"} for call in value]
            finish = choice["finish_reason"]
        return {
            "id": chunk["id"], "object": "chat.completion", "created": chunk["created"],
            "model": self.name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        }


def handler_for(chat):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, {"status": "ok", "model": chat.name, "context": chat.context})
            if self.path == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [{"id": chat.name, "object": "model"}]})
            return self.send_json(404, {"error": {"message": "Unknown endpoint"}})

        def do_POST(self):
            status = 200
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if self.path == "/apply-template":
                    payload = {"prompt": chat.render(request)}
                elif self.path == "/tokenize":
                    payload = {"tokens": chat.tokenize(request["content"], request.get("add_special", False))}
                elif self.path == "/v1/chat/completions":
                    if request.get("stream"):
                        completion = chat.prepare(request)
                        options = request.get("stream_options") or {}
                        if not isinstance(options, dict):
                            raise ValueError("stream_options must be an object")
                        return self.send_stream(completion, bool(options.get("include_usage")))
                    payload = chat.complete(request)
                else:
                    status, payload = 404, {"error": {"message": "Unknown endpoint"}}
            except (KeyError, TypeError, ValueError) as error:
                status, payload = 400, {"error": {"message": str(error)}}
            except (FloatingPointError, RuntimeError) as error:
                status, payload = 500, {"error": {"message": str(error)}}
            try:
                self.send_json(status, payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_stream(self, completion, include_usage):
            def event(data):
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                with closing(chat.chunks(completion)) as chunks:
                    try:
                        for chunk in chunks:
                            if "usage" not in chunk or include_usage:
                                event(json.dumps(chunk, ensure_ascii=False))
                    except (FloatingPointError, RuntimeError, ValueError) as error:
                        event(json.dumps({"error": {"message": str(error)}}))
                event("[DONE]")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return Handler


def serve(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((args.model / "generation_config.json").read_text(encoding="utf-8"))
    eos = config.get("eos_token_id", [])
    eos = {eos} if isinstance(eos, int) else set(eos)
    with Cuda(args.device) as cuda:
        model = DeviceModel(args.model, cuda)
        try:
            drafter = None
            if args.drafter:
                from .drafter import Drafter

                drafter = Drafter(args.drafter, model, args.context, f"cuda:{args.device}")
            chat = Chat(tokenizer, model, eos, args.context, args.name or args.model.name, args.no_thinking,
                        drafter=drafter, draft_block_size=args.draft_block_size or 2, device=args.device)
            # Keep request handling on the thread where the CUDA context is current.
            with HTTPServer(("127.0.0.1", args.port), handler_for(chat)) as server:
                print(json.dumps({"status": "ok", "model": chat.name, "context": chat.context,
                                  "port": server.server_port, "no_thinking": args.no_thinking,
                                  "draft_block_size": chat.draft_block_size if drafter is not None else None}), flush=True)
                server.serve_forever()
        finally:
            model.close()
