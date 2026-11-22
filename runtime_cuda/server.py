"""Inkling CUDA inference behind the shared OpenAI-compatible HTTP transport."""
from contextlib import ExitStack, closing
import threading

from . import eos_ids, load_drafter, open_model, stream_tokens
from runtime_mlx.server import ClientGone, EFFORTS, OutputParser, RequestError, SPECIAL, normalize_messages, request_tools


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
            self.ids = {}
            for key, special in SPECIAL.items():
                tid = self.tokenizer.convert_tokens_to_ids(special)
                if not isinstance(tid, int) or tid < 0:
                    raise ValueError(f"tokenizer has no {special}; is this an Inkling-2 model?")
                self.ids[key] = tid
            self.eos = eos_ids(self.model, self.tokenizer) | {self.ids["eos"]}
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
        messages = normalize_messages(req.get("messages"))
        tools, tool_names = request_tools(req)
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
        hard_no_thinking = getattr(self.args, "no_thinking", False) or enable is False
        if hard_no_thinking:
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
        return rendered, tool_names, hard_no_thinking

    def render(self, req):
        return self._render(req)[0]

    def prepare(self, req):
        rendered, tool_names, hard_no_thinking = self._render(req, generation=True)
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
        return {
            "ids": ids, "max_tokens": min(limit, remaining), "temperature": temperature, "seed": seed,
            "blocked_token": self.ids["thinking"] if hard_no_thinking else None,
            "tool_names": tool_names,
        }

    def run(self, job, emit, gone):
        parser = OutputParser(self.tokenizer, self.ids, job["tool_names"])
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
                    # A model may configure additional end-of-turn tokens.
                    for event in parser.feed(self.ids["eos"] if token in self.eos else token):
                        emit(*event)
                    if parser.done or count >= job["max_tokens"]:
                        break
            for event in parser.close():
                emit(*event)
            if parser.done:
                finish = parser.finish
        return {"prompt_tokens": len(ids), "completion_tokens": count, "total_tokens": len(ids) + count,
                "prompt_tokens_details": {"cached_tokens": 0}}, finish

    def warmup(self):
        for temperature in (0., 0.7):
            job = self.prepare({"messages": [{"role": "user", "content": "Say OK."}],
                                "max_tokens": 4, "temperature": temperature})
            self.run(job, lambda *_: None, lambda: False)
