"""Inkling CUDA inference behind the shared OpenAI-compatible HTTP transport."""
from contextlib import ExitStack, closing
import threading
from uuid import uuid4

from . import eos_ids, load_drafter, open_model, stream_tokens
from .inkling_cuda.serve import END, MODEL, TEXT, THINK, TOOL, Frames


class Engine:
    def __init__(self, args):
        if args.top_p != 1.:
            raise ValueError("CUDA serving supports temperature and seed, with top_p fixed at 1")
        self.args = args
        self.lock = threading.Lock()
        self.stack = ExitStack()
        try:
            self.model, self.tokenizer = self.stack.enter_context(
                open_model(args.model, device=args.device, engine=getattr(args, "engine", "auto"),
                           revision=getattr(args, "revision", None), drafter=args.drafter)
            )
            if not 1 <= args.context <= self.model.context_limit:
                raise ValueError(f"CUDA context must be within [1, {self.model.context_limit}]")
            self.eos = eos_ids(self.model, self.tokenizer)
            self.end, self.message, self.text, self.thinking, self.tool = self.tokenizer.convert_tokens_to_ids(
                [END, MODEL, TEXT, THINK, TOOL]
            )
            self.drafter = load_drafter(self.model, args.drafter, context=args.context) if args.drafter else None
            self.draft_block_size = getattr(args, "draft_block_size", None) or 2
        except BaseException:
            self.stack.close()
            raise

    def close(self):
        with self.lock:
            self.stack.close()

    def tokenize(self, content, add_special=False):
        return self.tokenizer.encode(content, add_special_tokens=add_special)

    def _render(self, req, *, generation=False):
        from runtime_mlx.server import EFFORTS, RequestError, normalize_messages

        messages = normalize_messages(req.get("messages"))
        tools = req.get("tools")
        if tools is not None and (not isinstance(tools, list) or any(
            not isinstance(tool, dict) or not isinstance(tool.get("function"), dict)
            or not isinstance(tool["function"].get("name"), str) for tool in tools
        )):
            raise RequestError(400, "tools must be a list of named functions")
        add_generation_prompt = req.get("add_generation_prompt", True)
        if type(add_generation_prompt) is not bool:
            raise RequestError(400, "add_generation_prompt must be a boolean")
        template = req.get("chat_template_kwargs") or {}
        if not isinstance(template, dict):
            raise RequestError(400, "chat_template_kwargs must be an object")
        effort = req.get("reasoning_effort", template.get("reasoning_effort"))
        if isinstance(effort, dict):
            effort = effort.get("effort")
        enable = template.get("enable_thinking")
        if enable is not None and type(enable) is not bool:
            raise RequestError(400, "enable_thinking must be a boolean")
        if getattr(self.args, "no_thinking", False) or enable is False:
            effort = "none"
        elif effort is None:
            effort = self.args.reasoning_effort
            if enable and effort == "none":
                effort = "low"
        if isinstance(effort, str):
            effort = effort.strip().lower()
            if effort not in EFFORTS:
                raise RequestError(400, "Unknown reasoning effort")
        elif type(effort) not in (float, int) or not 0 <= effort <= 0.99:
            raise RequestError(400, "Reasoning effort must be in [0, 0.99]")
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tools=tools or None, tokenize=False, reasoning_effort=effort,
                add_generation_prompt=generation or add_generation_prompt,
            )
        except Exception as error:
            raise RequestError(400, f"chat template error: {error}") from error
        return rendered, effort, messages

    def render(self, req):
        return self._render(req)[0]

    def prepare(self, req):
        from runtime_mlx.server import RequestError

        rendered, effort, messages = self._render(req, generation=True)
        ids = self.tokenize(rendered)
        remaining = self.args.context - len(ids)
        if not ids or remaining < 1:
            raise RequestError(400, "The request exceeds the available context size", "exceed_context_size_error")
        limit = req.get("max_completion_tokens", req.get("max_tokens"))
        if limit is None:
            limit = self.args.max_tokens
        if type(limit) is not int or limit < 1:
            raise RequestError(400, "Completion limit must be a positive integer")
        temperature = req.get("temperature")
        if temperature is None:
            temperature = self.args.temp
        if type(temperature) not in (float, int) or not 0 <= temperature <= 2:
            raise RequestError(400, "Temperature must be in [0, 2]")
        if req.get("top_p", self.args.top_p) not in (None, 1.) or req.get("top_k") not in (None, 0):
            raise RequestError(400, "CUDA sampling supports temperature and seed")
        seed = req.get("seed")
        if seed is not None and (type(seed) is not int or not -(1 << 63) <= seed < (1 << 64)):
            raise RequestError(400, "Seed must be a 64-bit integer")
        tools = req.get("tools") or []
        return {
            "ids": ids, "max_tokens": min(limit, remaining), "temperature": temperature, "seed": seed,
            "blocked_token": self.thinking if effort in ("none", 0) else None,
            "tools": tools, "stop_at_text_end": not tools or messages[-1]["role"] == "tool",
        }

    @staticmethod
    def _emit(emit, delta):
        for key, value in delta.items():
            if key == "tool_calls":
                for call in value:
                    emit("tool", {k: v for k, v in call.items() if k != "index"})
            else:
                emit("reasoning" if key == "reasoning_content" else key, value)

    def run(self, job, emit, gone):
        from runtime_mlx.server import ClientGone

        frames = Frames(self, job["tools"], uuid4().hex)
        ids = job["ids"]
        count, finish = 0, "length"
        with self.lock:
            if gone():
                raise ClientGone()
            with closing(stream_tokens(self.model, ids, max_tokens=job["max_tokens"], eos_ids=self.eos,
                                       blocked_token=job["blocked_token"], drafter=self.drafter,
                                       draft_block_size=self.draft_block_size, temperature=job["temperature"],
                                       seed=job["seed"])) as tokens:
                for count, token in enumerate(tokens, 1):
                    if gone():
                        raise ClientGone()
                    if token in self.eos:
                        finish = "stop"
                        break
                    delta, stop = frames.token(token)
                    self._emit(emit, delta)
                    if stop and (stop == "tool_calls" or job["stop_at_text_end"] or frames.channel == "tool"):
                        finish = stop
                        break
            self._emit(emit, frames.finish())
        return {"prompt_tokens": len(ids), "completion_tokens": count, "total_tokens": len(ids) + count}, finish

    def warmup(self):
        for temperature in (0., 0.7):
            job = self.prepare({"messages": [{"role": "user", "content": "Say OK."}],
                                "max_tokens": 4, "temperature": temperature})
            self.run(job, lambda *_: None, lambda: False)
