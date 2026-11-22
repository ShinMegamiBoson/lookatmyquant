"""Loopback OpenAI-compatible chat and tokenizer endpoints for Inkling CUDA."""
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


def render_chat(tokenizer, request, *, no_thinking):
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
        reasoning_effort="none" if no_thinking else request.get("reasoning_effort", "none"),
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


class Chat:
    def __init__(self, tokenizer, model, eos, context, name, no_thinking):
        self.tokenizer = tokenizer
        self.model = model
        self.eos = eos
        self.context = context
        self.name = name
        self.no_thinking = no_thinking
        self.end, self.tool, thinking = tokenizer.convert_tokens_to_ids([END, TOOL, THINK])
        self.blocked = thinking if no_thinking else None
        self.calls = 0

    def render(self, request):
        return render_chat(self.tokenizer, request, no_thinking=self.no_thinking)

    def tokenize(self, text, add_special=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special)

    def complete(self, request):
        if request.get("stream"):
            raise ValueError("Streaming is unavailable")
        if request.get("model", self.name) != self.name:
            raise ValueError("Unknown model")
        limit = request.get("max_completion_tokens", request.get("max_tokens", 512))
        if type(limit) is not int or limit < 1:
            raise ValueError("Completion limit must be a positive integer")
        prompt = self.tokenize(self.render(request))
        if not prompt or len(prompt) + limit > self.context:
            raise ValueError("Prompt and completion allowance exceed the configured context")
        output = []
        stop = False
        inside_tool = False
        with self.model.session(self.context) as session:
            session.prefill(prompt[:-1])
            token = prompt[-1]
            for _ in range(limit):
                token = session.predict(token, blocked_token=self.blocked)
                output.append(token)
                if token in self.eos or token == self.end and inside_tool:
                    stop = True
                    break
                if token == self.tool:
                    inside_tool = True
        visible = output[:-1] if output[-1] in self.eos else output
        completion_id = f"inkling_{self.calls}"
        self.calls += 1
        message, finish_reason = assistant_message(
            self.tokenizer.decode(visible, skip_special_tokens=False), request.get("tools"), completion_id)
        return {
            "id": f"chatcmpl-{completion_id}", "object": "chat.completion", "created": int(time.time()),
            "model": self.name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason if stop else "length"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(output),
                      "total_tokens": len(prompt) + len(output)},
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

    return Handler


def serve(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((args.model / "generation_config.json").read_text(encoding="utf-8"))
    eos = config.get("eos_token_id", [])
    eos = {eos} if isinstance(eos, int) else set(eos)
    with Cuda(args.device) as cuda:
        model = DeviceModel(args.model, cuda)
        chat = Chat(tokenizer, model, eos, args.context, args.name or args.model.name, args.no_thinking)
        try:
            # Keep request handling on the thread where the CUDA context is current.
            with HTTPServer(("127.0.0.1", args.port), handler_for(chat)) as server:
                print(json.dumps({"status": "ok", "model": chat.name, "context": chat.context,
                                  "port": server.server_port, "no_thinking": args.no_thinking}), flush=True)
                server.serve_forever()
        finally:
            model.close()
