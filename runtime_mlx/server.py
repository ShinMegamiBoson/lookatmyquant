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

* By default (`--max-batch 32`), packed Inkling-2-Mini models on MLX use continuous batching with
  paged KV pools, automatic zero-copy shared-prefix reuse, interleaved chunked prefill, and
  load-adaptive DSpark (`--dspark-max-batch 4`). Pass `--max-batch 1` for single-stream serving
  (concurrent requests receive HTTP 503 when all admission slots are full).

With `--ui`, the chat page at `GET /` can send same-origin requests over loopback HTTP or
through a Host-preserving HTTPS proxy in front of an authenticated network listener.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import math
import os
import re
import select
import socket
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

MAX_REQUEST_BYTES = 1 << 20
MAX_CONNECTIONS = 8
REQUEST_TIMEOUT = 15
UI_PAGE = Path(__file__).with_name("chat.html")
UI_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}
_HTTP_AUTHORITY = re.compile(r"(?:[A-Za-z0-9._-]+|\[[A-Fa-f0-9:.]+\])(?::[0-9]+)?")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _number(req, name, default, *, low, high=math.inf, integer=False):
    val = req.get(name)
    if val is None:
        return default
    valid_type = type(val) is int if integer else type(val) in (int, float)
    if not valid_type or not low <= val <= high or val == math.inf:
        raise RequestError(400, f"invalid {name}")
    return val


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


def request_tools(req: dict) -> tuple[list, set[str]]:
    """Select the tools shown to the model and the names accepted in its response."""
    if req.get("functions") is not None or req.get("function_call") is not None:
        raise RequestError(400, "use tools and tool_choice for function calls")
    tools = req.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise RequestError(400, "tools must be an array")
    tool_names: set[str] = set()
    for tool in tools or ():
        fn = tool.get("function") if isinstance(tool, dict) and tool.get("type") == "function" else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            raise RequestError(400, "each tool must declare a named function")
        tool_names.add(name)
    choice = req.get("tool_choice")
    if choice == "none":
        return [], set()
    if choice is not None and choice != "auto":
        raise RequestError(400, "supported tool_choice values are 'auto' and 'none'")
    return tools or [], tool_names


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
                    part = p.get("text")
                    if not isinstance(part, str):
                        raise RequestError(400, "text content parts require a string")
                    parts.append(part)
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
            raw_calls = m.get("tool_calls")
            if raw_calls is not None and not isinstance(raw_calls, list):
                raise RequestError(400, "tool_calls must be an array")
            for tc in raw_calls or []:
                if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                    raise RequestError(400, "each tool call requires a function object")
                fn = tc["function"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except (ValueError, RecursionError):
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


_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
REPEAT_GUARD_MIN_STREAK = 6
REPEAT_GUARD_MAX_RESULT_CHARS = 200


def _canon_tool_args(args: Any) -> str:
    """Canonical argument signature ignoring digits and whitespace (Kintsugi rule 1)."""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except (ValueError, RecursionError):
            pass
    raw = json.dumps(args, sort_keys=True, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
    return _WS_RE.sub(" ", _DIGITS_RE.sub("", raw)).strip()


def apply_repetition_guard(
    messages: list[dict],
    *,
    endpoint: str = "chat",
    min_streak: int = REPEAT_GUARD_MIN_STREAK,
    max_result_chars: int = REPEAT_GUARD_MAX_RESULT_CHARS,
) -> tuple[list[dict], dict | None]:
    """Detect stuck agent tool loops and append a single transient nudge for this request only.

    Triggers when the last `min_streak` assistant turns each made one call to the same tool with
    near-identical arguments (ignoring digits) and received short tool results (<= `max_result_chars`).
    """
    if len(messages) < 2 * min_streak:
        return messages, None
    streak = 0
    tool_name: str | None = None
    arg_sig: str | None = None
    i = len(messages) - 1
    while i >= 1:
        m_tool = messages[i]
        m_asst = messages[i - 1]
        if m_tool.get("role") != "tool" or m_asst.get("role") != "assistant":
            break
        calls = m_asst.get("tool_calls") or []
        if len(calls) != 1:
            break
        res_text = str(m_tool.get("content") or "").strip()
        if len(res_text) > max_result_chars:
            break
        fn = calls[0].get("function") or {}
        name = str(fn.get("name") or "")
        if not name:
            break
        sig = _canon_tool_args(fn.get("arguments", {}))
        if tool_name is None:
            tool_name, arg_sig = name, sig
        elif name != tool_name or sig != arg_sig:
            break
        streak += 1
        i -= 2
    if streak < min_streak or tool_name is None:
        return messages, None
    note = (
        f"You have made nearly the same {tool_name} call {streak} times in a row. "
        "Try a different step or give your answer."
    )
    info = {"tool": tool_name, "streak": streak, "endpoint": endpoint}
    log(f"WARN repetition guard ({endpoint}): {streak} calls to {tool_name!r} in a row")
    return [*messages, {"role": "user", "content": note}], info


def _anthropic_text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for blk in content:
        if isinstance(blk, str):
            parts.append(blk)
        elif isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(str(blk.get("text", "")))
        elif isinstance(blk, dict) and blk.get("type") == "image":
            parts.append("[image omitted: this server is text-only]")
    return "".join(parts)


def anthropic_to_openai(req: dict) -> dict:
    """Translate an Anthropic `/v1/messages` request into an OpenAI-shaped request dict."""
    if not isinstance(req, dict):
        raise RequestError(400, "request body must be a JSON object")
    raw_msgs = req.get("messages")
    if not isinstance(raw_msgs, list) or not raw_msgs:
        raise RequestError(400, "messages must be a non-empty list")
    out_msgs: list[dict[str, Any]] = []
    if req.get("system") is not None:
        sys_text = _anthropic_text_blocks(req["system"])
        if sys_text:
            out_msgs.append({"role": "system", "content": sys_text})
    for m in raw_msgs:
        if not isinstance(m, dict):
            raise RequestError(400, "each message must be an object")
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            raise RequestError(400, f"unsupported Anthropic role: {role!r}")
        if isinstance(content, str) or content is None:
            out_msgs.append({"role": role, "content": content or ""})
            continue
        if not isinstance(content, list):
            raise RequestError(400, "message content must be a string or block array")
        if role == "assistant":
            texts, thinks, calls = [], [], []
            for blk in content:
                if not isinstance(blk, dict):
                    raise RequestError(400, "content blocks must be objects")
                btype = blk.get("type")
                if btype == "text":
                    texts.append(str(blk.get("text", "")))
                elif btype == "thinking":
                    thinks.append(str(blk.get("thinking", "")))
                elif btype == "tool_use":
                    name = blk.get("name")
                    if not isinstance(name, str) or not name:
                        raise RequestError(400, "tool_use block requires a name")
                    inp = blk.get("input")
                    if not isinstance(inp, dict):
                        inp = {} if inp is None else {"input": inp}
                    calls.append({
                        "id": blk.get("id") or ("call_" + uuid.uuid4().hex[:24]),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(inp, ensure_ascii=False)},
                    })
            am: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinks:
                am["reasoning_content"] = "\n\n".join(t for t in thinks if t)
            if calls:
                am["tool_calls"] = calls
            out_msgs.append(am)
        else:
            # User turn: Anthropic sends `tool_result` blocks inside user messages; split into tool + user turns.
            user_parts: list[str] = []
            for blk in content:
                if not isinstance(blk, dict):
                    raise RequestError(400, "content blocks must be objects")
                btype = blk.get("type")
                if btype == "tool_result":
                    if user_parts:
                        out_msgs.append({"role": "user", "content": "".join(user_parts)})
                        user_parts = []
                    out_msgs.append({
                        "role": "tool",
                        "tool_call_id": str(blk.get("tool_use_id") or ""),
                        "content": _anthropic_text_blocks(blk.get("content", "")),
                    })
                elif btype == "text":
                    user_parts.append(str(blk.get("text", "")))
                elif btype == "image":
                    user_parts.append("[image omitted: this server is text-only]")
            if user_parts or not out_msgs or out_msgs[-1]["role"] != "tool":
                out_msgs.append({"role": "user", "content": "".join(user_parts)})

    oai: dict[str, Any] = {"messages": out_msgs, "_endpoint": "messages"}
    if "max_tokens" in req:
        oai["max_tokens"] = req["max_tokens"]
    if "temperature" in req:
        oai["temperature"] = req["temperature"]
    if "top_p" in req:
        oai["top_p"] = req["top_p"]
    if "top_k" in req:
        oai["top_k"] = req["top_k"]
    if "stream" in req:
        oai["stream"] = req["stream"]
    if "cache_prompt" in req:
        oai["cache_prompt"] = req["cache_prompt"]

    # Map Anthropic thinking / output_config.effort -> reasoning_effort
    effort = req.get("reasoning_effort")
    if effort is None and isinstance(req.get("output_config"), dict):
        effort = req["output_config"].get("effort")
    thinking = req.get("thinking")
    if effort is None and isinstance(thinking, dict):
        ttype = thinking.get("type")
        if ttype == "disabled":
            effort = "none"
        elif ttype == "enabled":
            budget = thinking.get("budget_tokens")
            if isinstance(budget, int):
                if budget <= 0:
                    effort = "none"
                elif budget <= 1024:
                    effort = "low"
                elif budget <= 4096:
                    effort = "medium"
                else:
                    effort = "high"
    if effort is not None:
        oai["reasoning_effort"] = effort

    if req.get("tools") is not None:
        if not isinstance(req["tools"], list):
            raise RequestError(400, "tools must be an array")
        tools = []
        for t in req["tools"]:
            if not isinstance(t, dict) or not isinstance(t.get("name"), str) or not t["name"]:
                raise RequestError(400, "each Anthropic tool requires a name")
            fn: dict[str, Any] = {"name": t["name"]}
            if t.get("description") is not None:
                fn["description"] = t["description"]
            if t.get("input_schema") is not None:
                fn["parameters"] = t["input_schema"]
            tools.append({"type": "function", "function": fn})
        oai["tools"] = tools

    tc = req.get("tool_choice")
    if tc is not None:
        if isinstance(tc, str):
            oai["tool_choice"] = tc
        elif isinstance(tc, dict):
            ttype = tc.get("type")
            if ttype in ("auto", "none"):
                oai["tool_choice"] = ttype
            else:
                raise RequestError(400, "supported tool_choice values are 'auto' and 'none'")
        else:
            raise RequestError(400, "supported tool_choice values are 'auto' and 'none'")
    return oai


def responses_to_openai(req: dict) -> dict:
    """Translate an OpenAI `/v1/responses` request (used by Codex CLI) into `/v1/chat/completions` shape."""
    if not isinstance(req, dict):
        raise RequestError(400, "request body must be a JSON object")
    out_msgs: list[dict[str, Any]] = []
    instructions = req.get("instructions")
    if isinstance(instructions, str) and instructions:
        out_msgs.append({"role": "system", "content": instructions})
    inp = req.get("input")
    if isinstance(inp, str):
        out_msgs.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        pending_reasoning: list[str] = []
        for item in inp:
            if isinstance(item, str):
                out_msgs.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                raise RequestError(400, "each input item must be an object or string")
            itype = item.get("type")
            if itype == "reasoning":
                parts = []
                for s in item.get("summary") or ():
                    if isinstance(s, dict) and isinstance(s.get("text"), str):
                        parts.append(s["text"])
                    elif isinstance(s, str):
                        parts.append(s)
                if isinstance(item.get("content"), str):
                    parts.append(item["content"])
                if parts:
                    pending_reasoning.append("\n".join(parts))
            elif itype == "function_call":
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    raise RequestError(400, "function_call item requires a name")
                call = {
                    "id": str(item.get("call_id") or item.get("id") or ("call_" + uuid.uuid4().hex[:24])),
                    "type": "function",
                    "function": {"name": name, "arguments": item.get("arguments", "{}")},
                }
                if out_msgs and out_msgs[-1]["role"] == "assistant" and not out_msgs[-1].get("tool_calls"):
                    out_msgs[-1]["tool_calls"] = [call]
                    if pending_reasoning:
                        prev = out_msgs[-1].get("reasoning_content", "")
                        out_msgs[-1]["reasoning_content"] = "\n\n".join(x for x in [prev, *pending_reasoning] if x)
                        pending_reasoning = []
                else:
                    am: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": [call]}
                    if pending_reasoning:
                        am["reasoning_content"] = "\n\n".join(pending_reasoning)
                        pending_reasoning = []
                    out_msgs.append(am)
            elif itype == "function_call_output":
                out_val = item.get("output", "")
                if not isinstance(out_val, str):
                    out_val = json.dumps(out_val, ensure_ascii=False)
                out_msgs.append({
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": out_val,
                })
            else:
                # Standard message item (`type: "message"` or bare `{"role": ..., "content": ...}`)
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    norm_parts = []
                    for p in content:
                        if isinstance(p, str):
                            norm_parts.append({"type": "text", "text": p})
                        elif isinstance(p, dict) and p.get("type") in (None, "text", "input_text", "output_text"):
                            norm_parts.append({"type": "text", "text": str(p.get("text", ""))})
                        elif isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"):
                            norm_parts.append({"type": "image"})
                    content = norm_parts
                msg: dict[str, Any] = {"role": role, "content": content}
                if role == "assistant" and pending_reasoning:
                    msg["reasoning_content"] = "\n\n".join(pending_reasoning)
                    pending_reasoning = []
                out_msgs.append(msg)
    else:
        raise RequestError(400, "input must be a string or list of items")
    if not out_msgs:
        raise RequestError(400, "input produced no messages")

    oai: dict[str, Any] = {"messages": out_msgs, "_endpoint": "responses"}
    if req.get("max_output_tokens") is not None:
        oai["max_tokens"] = req["max_output_tokens"]
    elif req.get("max_tokens") is not None:
        oai["max_tokens"] = req["max_tokens"]
    for k in ("temperature", "top_p", "top_k", "stream", "cache_prompt", "seed"):
        if k in req:
            oai[k] = req[k]
    if isinstance(req.get("reasoning"), dict) and req["reasoning"].get("effort") is not None:
        oai["reasoning_effort"] = req["reasoning"]["effort"]
    elif req.get("reasoning_effort") is not None:
        oai["reasoning_effort"] = req["reasoning_effort"]

    if req.get("tools") is not None:
        if not isinstance(req["tools"], list):
            raise RequestError(400, "tools must be an array")
        tools = []
        for t in req["tools"]:
            if not isinstance(t, dict):
                raise RequestError(400, "each tool must be an object")
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                tools.append(t)
            elif t.get("type") == "function" and isinstance(t.get("name"), str) and t["name"]:
                fn: dict[str, Any] = {"name": t["name"]}
                if t.get("description") is not None:
                    fn["description"] = t["description"]
                if t.get("parameters") is not None:
                    fn["parameters"] = t["parameters"]
                tools.append({"type": "function", "function": fn})
            else:
                raise RequestError(400, "each tool must declare a named function")
        oai["tools"] = tools

    if req.get("tool_choice") is not None:
        oai["tool_choice"] = req["tool_choice"]
    return oai


# --------------------------------------------------------------------------- output parsing


def _reject_json_constant(value: str):
    raise ValueError(f"{value} is not a JSON number")


class OutputParser:
    """Token-level parser for Inkling's output grammar.

        <|message_model|><|content_thinking|>...<|end_message|>                -> reasoning
        <|message_model|><|content_text|>...<|end_message|>                    -> content
        <|message_model|>NAME<|content_invoke_tool_json|>{json}<|end_message|> -> one tool call
        <|content_model_end_sampling|>                                         -> end of turn

    Parsing stops after the first complete call to a declared function. Invalid tool
    frames are skipped so later assistant text can still be returned. `feed()` returns events `("reasoning"|"content", text)` and
    `("tool", openai_tool_call)`.
    """

    def __init__(self, tokenizer, ids: dict[str, int], tool_names=()):
        self.tok = tokenizer
        self.tool_names = frozenset(tool_names)
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
        try:
            obj = json.loads(self._dec(self.tool_tokens), parse_constant=_reject_json_constant)
            if not isinstance(obj, dict):
                return
            name = self.tool_name
            if not name or ("name" in obj and ("args" in obj or "arguments" in obj)):
                name = obj.get("name")
                if self.tool_name and name != self.tool_name:
                    return
                args = obj.get("args", obj.get("arguments"))
            else:
                args = obj.get("arguments", obj)
            if not isinstance(name, str) or name not in self.tool_names or not isinstance(args, dict):
                return
            arguments = json.dumps(args, ensure_ascii=False, allow_nan=False)
            arguments.encode("utf-8")  # Reject unpaired surrogates before writing SSE.
        except (ValueError, RecursionError):
            return
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
                self.state = "tool" if t == I["tool_json"] and self.tool_names else "discard_tool"
                self.tool_tokens = []
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
        elif st in ("tool", "discard_tool"):
            if t == I["end"]:
                if st == "tool":
                    self._finish_tool(out)
                self.state = "between"
            elif t == I["model"]:
                self.state, self.header = "header", []
            elif t in (I["text"], I["thinking"]):
                self._enter("thinking" if t == I["thinking"] else "text")
            elif st == "tool":
                self.tool_tokens.append(t)
        return out

    def close(self) -> list:
        out: list = []
        if self.state in ("thinking", "text"):
            self._flush(out)
        return out


# --------------------------------------------------------------------------- engine


class Engine:
    """Loads the model once and serves either continuous-batched (`--max-batch > 1`) or single-stream requests."""

    def __init__(self, args: argparse.Namespace):
        # MLX arrays carry stream handles that are local to their creating thread.
        # Load and execute on one persistent thread across all HTTP connections.
        self.scheduler = None
        self.max_concurrent = 1
        self._sched_future = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lamq-mlx")
        try:
            self._executor.submit(self._load, args).result()
            if self.scheduler is not None:
                self._sched_future = self._executor.submit(self.scheduler.run_forever)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        sched = getattr(self, "scheduler", None)
        if sched is not None:
            sched.stop()
        sf = getattr(self, "_sched_future", None)
        if sf is not None:
            try:
                sf.result(timeout=5.0)
            except Exception:
                pass
            self._sched_future = None
        ex = getattr(self, "_executor", None)
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)

    def _load(self, args: argparse.Namespace) -> None:
        import mlx.core as mx

        from .inkling import ContinuousBatchScheduler, PromptCache
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
        cache_dir = getattr(args, "prompt_cache_dir", None)
        req_max_batch = max(1, int(getattr(args, "max_batch", 32)))
        if req_max_batch > 1 and cache_dir:
            log("note: falling back to --max-batch 1 because --prompt-cache-dir is active")
            req_max_batch = 1
        if req_max_batch > 1 and self.m.cfg.use_fast_metal:
            self.prompt_cache = None
            allow_pfx = (
                not getattr(args, "no_prefix_cache", False)
                and _is_loopback(getattr(args, "host", "127.0.0.1"))
                and not os.environ.get("LAMQ_API_KEY")
            )
            self.scheduler = ContinuousBatchScheduler(
                self.m,
                drafter=self.drafter,
                eos_id=self.ids["eos"],
                max_batch=req_max_batch,
                prefill_step=max(64, int(getattr(args, "prefill_step", 512))),
                dspark_max_batch=max(0, int(getattr(args, "dspark_max_batch", 4))),
                enable_prefix_cache=allow_pfx,
                auto_start=False,
            )
            self.max_concurrent = req_max_batch * 2
        else:
            self.scheduler = None
            self.max_concurrent = 1
            if cache_dir:
                from .prompt_cache_disk import DiskPromptCache

                self.prompt_cache = DiskPromptCache(
                    cache_dir,
                    max_entries=max(1, int(args.prompt_cache or 2)),
                    max_disk_gb=float(getattr(args, "prompt_cache_gb", 20.0)),
                )
            elif args.prompt_cache:
                self.prompt_cache = PromptCache(max_entries=args.prompt_cache)
            else:
                self.prompt_cache = None
        batch_note = f", max_batch={req_max_batch}" if self.scheduler is not None else ", single-stream"
        log(f"loaded {args.model} ({'with drafter' if self.drafter is not None else 'serial'}{batch_note}) in "
            f"{time.perf_counter() - t:.1f}s, {mx.get_active_memory() / 1e9:.2f} GB")

    def _repeat_guard_enabled(self) -> bool:
        if getattr(self.args, "no_repeat_guard", False):
            return False
        return os.environ.get("LAMQ_REPEAT_GUARD", "1").strip() not in ("0", "false", "no", "off")

    def _apply_template(self, req: dict, *, tokenize: bool, guard: bool = False):
        messages = normalize_messages(req.get("messages"))
        if guard and self._repeat_guard_enabled():
            messages, _ = apply_repetition_guard(messages, endpoint=str(req.get("_endpoint", "chat")))
        tools, tool_names = request_tools(req)
        effort = req.get("reasoning_effort")
        if isinstance(effort, dict):
            effort = effort.get("effort")
        if isinstance(effort, str):
            effort = effort.strip().lower()
            if effort not in EFFORTS:
                raise RequestError(400, "unknown reasoning_effort")
        else:
            effort = _number({"reasoning_effort": effort}, "reasoning_effort", self.args.reasoning_effort, low=0, high=0.99)
        add_prompt = req.get("add_generation_prompt", True)
        if type(add_prompt) is not bool:
            raise RequestError(400, "add_generation_prompt must be a boolean")
        kw: dict[str, Any] = dict(tokenize=tokenize, add_generation_prompt=add_prompt, reasoning_effort=effort)
        if tools:
            kw["tools"] = tools
        try:
            return self.tok.apply_chat_template(messages, **kw), effort, tool_names
        except Exception as e:  # jinja raise_exception() and friends
            raise RequestError(400, "chat template could not render messages and tools") from e

    def render(self, req: dict) -> str:
        return self._apply_template(req, tokenize=False, guard=False)[0]

    def tokenize(self, content: str, add_special: bool = False) -> list[int]:
        return self.tok.encode(content, add_special_tokens=add_special)

    def prepare(self, req: dict) -> dict:
        """Validate and tokenize a request before sending any response bytes."""
        enc, effort, tool_names = self._apply_template(dict(req, add_generation_prompt=True), tokenize=True, guard=True)
        if isinstance(enc, dict) or hasattr(enc, "input_ids"):
            enc = enc["input_ids"]
        ids = [int(x) for x in enc]
        ctx = self.args.context
        if not ids:
            raise RequestError(400, "chat template produced no tokens")
        if len(ids) >= ctx:
            raise RequestError(400, f"the request exceeds the available context size, try increasing it "
                                    f"(prompt {len(ids)} tokens, context {ctx})", "exceed_context_size_error")
        max_tokens = _number(req, "max_completion_tokens", None, low=1, integer=True)
        if max_tokens is None:
            max_tokens = _number(req, "max_tokens", self.args.max_tokens, low=1, integer=True)
        return {
            "ids": ids,
            "effort": effort,
            "tool_names": tool_names,
            "max_tokens": min(max_tokens, ctx - len(ids)),
            "temp": _number(req, "temperature", self.args.temp, low=0, high=2),
            "top_p": _number(req, "top_p", self.args.top_p, low=0, high=1),
            "top_k": min(_number(req, "top_k", 0, low=0, integer=True), self.m.lm_head.order_mx.size),
            "use_cache": req.get("cache_prompt", True) is not False,
        }

    def run(self, job: dict, emit: Callable[[str, Any], None], gone: Callable[[], bool]) -> tuple[dict, str]:
        """Generate for a prepared request, sending parsed events to `emit`. Returns `(usage, finish_reason)`."""
        if getattr(self, "scheduler", None) is not None:
            return self._run_batched(job, emit, gone)
        return self._executor.submit(self._run, job, emit, gone).result()

    def _run_batched(self, job: dict, emit: Callable[[str, Any], None], gone: Callable[[], bool]) -> tuple[dict, str]:
        import queue as _queue

        ids = job["ids"]
        if gone():
            raise ClientGone()
        t0 = time.perf_counter()
        aborted = threading.Event()

        def _is_gone() -> bool:
            if aborted.is_set():
                return True
            if gone():
                aborted.set()
                return True
            return False

        tok_q = self.scheduler.submit(job, _is_gone)
        parser = OutputParser(self.hf, self.ids, job["tool_names"])
        n_gen = 0
        finish: str | None = None
        reused = 0
        prompt_s = 0.0
        try:
            while True:
                try:
                    kind, payload = tok_q.get(timeout=0.5)
                except _queue.Empty:
                    if _is_gone():
                        raise ClientGone()
                    continue
                if kind == "err":
                    raise payload
                if kind == "meta":
                    reused = int(payload.get("prompt_cached_tokens", 0) or 0)
                    prompt_s = float(payload.get("prefill_s", 0.0) or 0.0)
                    continue
                if kind == "done":
                    if payload.get("aborted"):
                        raise ClientGone()
                    if "prompt_cached_tokens" in payload:
                        reused = int(payload.get("prompt_cached_tokens", 0) or 0)
                    if "prefill_s" in payload:
                        prompt_s = float(payload.get("prefill_s", 0.0) or 0.0)
                    break
                n_gen += 1
                for ev in parser.feed(int(payload)):
                    emit(*ev)
                if parser.done or n_gen >= job["max_tokens"]:
                    aborted.set()
                    break
            for ev in parser.close():
                emit(*ev)
            finish = parser.finish if parser.done else "length"
        finally:
            aborted.set()
            elapsed = time.perf_counter() - t0
            log(f"[batch] prompt {len(ids)} tok (cached {reused}, prefill {prompt_s:.1f}s) | gen {n_gen} tok "
                f"({n_gen / max(elapsed - prompt_s, 1e-9):.1f} tok/s) | effort {job['effort']} temp {job['temp']} | "
                f"{finish or 'aborted'}")
        usage = {"prompt_tokens": len(ids), "completion_tokens": n_gen, "total_tokens": len(ids) + n_gen,
                 "prompt_tokens_details": {"cached_tokens": reused}}
        return usage, finish

    def _run(self, job: dict, emit: Callable[[str, Any], None], gone: Callable[[], bool]) -> tuple[dict, str]:
        import mlx.core as mx

        from .inkling import stream_dspark, stream_serial

        ids = job["ids"]
        if gone():
            raise ClientGone()
        stats: dict[str, Any] = {}
        common = dict(max_tokens=job["max_tokens"], eos_ids={self.ids["eos"]}, temp=job["temp"], top_p=job["top_p"],
                      top_k=job["top_k"], stats_out=stats, prefill_chunk=self.args.prefill_chunk,
                      prompt_cache=self.prompt_cache if job["use_cache"] else None)
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        if self.drafter is not None:
            gen = stream_dspark(self.m, self.drafter, ids, **common)
        else:
            gen = stream_serial(self.m, ids, **common)
        parser = OutputParser(self.hf, self.ids, job["tool_names"])
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


class BoundedHTTPServer(ThreadingHTTPServer):
    """Bound the number of HTTP worker threads, including connections still sending headers."""

    def __init__(self, address, handler, *, max_connections: int = MAX_CONNECTIONS):
        self._connections = threading.BoundedSemaphore(max(MAX_CONNECTIONS, int(max_connections)))
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


def make_handler(engine, model_id: str, *, api_key: str | None = None, loopback: bool = True, ui: bool = False):
    """Bind HTTP transport to an engine with `prepare`, `run` and `args.context`.

    `ui` serves the chat page at `GET /` and admits that page's same-origin requests.
    """
    if not loopback and not api_key:
        raise ValueError("network listening requires an API key")
    page = UI_PAGE.read_bytes() if ui else b""
    max_concurrent = max(1, int(getattr(engine, "max_concurrent", 1)))
    generation = threading.BoundedSemaphore(max_concurrent)
    expected_auth = f"Bearer {api_key}".encode() if api_key else None
    expected_xkey = api_key.encode() if api_key else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "lamq-serve"
        timeout = REQUEST_TIMEOUT

        def handle(self):
            # One request per connection, with a deadline for headers and body.
            # Generation has its own token budget and socket write timeout.
            def expire():
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            deadline = self._deadline = threading.Timer(REQUEST_TIMEOUT, expire)
            deadline.daemon = True
            deadline.start()
            try:
                self.handle_one_request()
            finally:
                deadline.cancel()
                self.close_connection = True

        def _authorized(self, require_key: bool = True) -> bool:
            hosts = self.headers.get_all("Host", [])
            origins = self.headers.get_all("Origin", [])
            if origins and not ui:
                self._error(403, "browser origins are disabled")
                return False

            if loopback or origins or not require_key:
                try:
                    if len(hosts) != 1 or not _HTTP_AUTHORITY.fullmatch(hosts[0]):
                        raise ValueError
                    authority = urlsplit("//" + hosts[0].lower())
                    hostname, port = authority.hostname, authority.port
                except ValueError:
                    self._error(403, "a valid Host is required")
                    return False
                local_host = _is_loopback(hostname)
                if loopback and not local_host:
                    self._error(403, "local requests require a loopback Host")
                    return False
                if origins:
                    # Compare the browser's serialized origin to the original Host at the TLS proxy.
                    # Default ports disappear during origin serialization. Forwarded headers are unnecessary.
                    scheme, default_port = ("http", 80) if local_host else ("https", 443)
                    host = f"[{hostname}]" if ":" in hostname else hostname
                    if port is not None and port != default_port:
                        host += f":{port}"
                    sites = self.headers.get_all("Sec-Fetch-Site", [])
                    if (len(origins) != 1 or origins[0].lower() != f"{scheme}://{host}"
                            or (sites and sites != ["same-origin"])):
                        self._error(403, "chat requests require the page's own origin")
                        return False
            if expected_auth is not None and require_key:
                auth = self.headers.get_all("Authorization", [])
                xkey = self.headers.get_all("x-api-key", [])
                ok_bearer = len(auth) == 1 and hmac.compare_digest(auth[0].encode(), expected_auth)
                ok_xkey = expected_xkey is not None and len(xkey) == 1 and hmac.compare_digest(xkey[0].encode(), expected_xkey)
                if not (ok_bearer or ok_xkey):
                    self._error(401, "valid bearer authentication required", "authentication_error")
                    return False
            return True

        def log_message(self, fmt, *a):
            pass

        def _json(self, status: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, etype: str = "invalid_request_error") -> None:
            self._json(status, {"error": {"message": message, "type": etype, "code": status}})

        def handle_expect_100(self):
            if self._authorized():
                self._error(417, "Expect is not supported")
            return False

        def _gone(self) -> bool:
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def do_OPTIONS(self):
            if self._authorized():
                self._error(405, "OPTIONS is not supported")

        def _page(self) -> None:
            body = page
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            for name, value in UI_HEADERS.items():
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if ui and path == "":
                # The page is static and holds no secrets; the API it calls still requires the key.
                if self._authorized(require_key=False):
                    self._page()
                return
            if not self._authorized():
                return
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model", "created": int(time.time()),
                                                             "owned_by": "lamq", "context_length": engine.args.context}]})
            else:
                self._error(404, f"not found: {self.path}")

        def do_POST(self):
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")
            utility = {"/apply-template": "render", "/tokenize": "tokenize"}.get(path)
            api_kind = {
                "/v1/chat/completions": "chat",
                "/chat/completions": "chat",
                "/v1/messages": "messages",
                "/messages": "messages",
                "/v1/responses": "responses",
                "/responses": "responses",
            }.get(path)
            if utility is not None:
                operation = getattr(engine, utility, None)
                if operation is None:
                    self._error(404, "endpoint not found")
                    return
            elif api_kind is None:
                self._error(404, "endpoint not found")
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get_all("Transfer-Encoding") or len(lengths) != 1:
                self._error(400, "one Content-Length header is required")
                return
            try:
                length = int(lengths[0])
            except ValueError:
                length = 0
            if length <= 0 or not lengths[0].isascii() or not lengths[0].isdigit():
                self._error(400, "invalid Content-Length")
                return
            if length > MAX_REQUEST_BYTES:
                self._error(413, "request body is too large")
                return
            if self.headers.get_content_type() != "application/json":
                self._error(415, "Content-Type must be application/json")
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request body")
                req = json.loads(raw)
                if not isinstance(req, dict):
                    raise ValueError("request body must be a JSON object")
            except (ValueError, RecursionError):
                self._error(400, "invalid JSON request body")
                return
            except OSError:
                return
            self._deadline.cancel()
            # A browser tab may still hold the conversation for a different model served here earlier.
            if utility is None and self.headers.get("Origin") and req.get("model", model_id) != model_id:
                self._error(409, "the served model has changed", "model_changed")
                return
            if not generation.acquire(blocking=False):
                self._error(503, "model is busy", "server_overloaded")
                return
            try:
                if utility is None:
                    self._generate(req, api_kind)
                else:
                    self._tokenizer_response(utility, operation, req)
            finally:
                generation.release()

        def _tokenizer_response(self, utility, operation, req):
            try:
                if utility == "render":
                    result = {"prompt": operation(req)}
                else:
                    content, add_special = req.get("content"), req.get("add_special", False)
                    if not isinstance(content, str) or type(add_special) is not bool:
                        raise RequestError(400, "tokenize requires string content and boolean add_special")
                    result = {"tokens": operation(content, add_special)}
            except RequestError as e:
                self._error(e.status, e.message, e.etype)
            except Exception:
                traceback.print_exc()
                self._error(500, "server error", "server_error")
            else:
                self._json(200, result)

        def _generate(self, req: dict, api_kind: str = "chat") -> None:
            try:
                if "stream" in req and type(req["stream"]) is not bool:
                    raise RequestError(400, "stream must be a boolean")
                if api_kind == "chat":
                    if req.get("stream_options") is not None and not isinstance(req["stream_options"], dict):
                        raise RequestError(400, "stream_options must be an object")
                    oai_req = req
                elif api_kind == "messages":
                    oai_req = anthropic_to_openai(req)
                else:
                    oai_req = responses_to_openai(req)
                job = engine.prepare(oai_req)
            except RequestError as e:
                self._error(e.status, e.message, e.etype)
                return
            except Exception:
                traceback.print_exc()
                self._error(500, "server error", "server_error")
                return
            streaming = bool(req.get("stream"))
            if api_kind == "chat":
                (self._stream if streaming else self._complete)(req, job)
            elif api_kind == "messages":
                (self._stream_anthropic if streaming else self._complete_anthropic)(req, job)
            else:
                (self._stream_responses if streaming else self._complete_responses)(req, job)

        def _complete(self, req: dict, job: dict) -> None:
            text = {"reasoning": [], "content": []}
            calls: list = []

            def emit(kind, payload):
                (calls if kind == "tool" else text[kind]).append(payload)

            try:
                usage, finish = engine.run(job, emit, self._gone)
            except ClientGone:
                return
            except Exception:
                traceback.print_exc()
                self._error(500, "server error", "server_error")
                return
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(text["content"]) or (None if calls else "")}
            if text["reasoning"]:
                msg["reasoning_content"] = "".join(text["reasoning"])
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                             "model": model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage})

        def _complete_anthropic(self, req: dict, job: dict) -> None:
            text = {"reasoning": [], "content": []}
            calls: list = []

            def emit(kind, payload):
                (calls if kind == "tool" else text[kind]).append(payload)

            try:
                usage, finish = engine.run(job, emit, self._gone)
            except ClientGone:
                return
            except Exception:
                traceback.print_exc()
                self._error(500, "server error", "server_error")
                return
            blocks: list[dict[str, Any]] = []
            if text["reasoning"]:
                blocks.append({"type": "thinking", "thinking": "".join(text["reasoning"])})
            body = "".join(text["content"])
            if body or (not blocks and not calls):
                blocks.append({"type": "text", "text": body})
            for tc in calls:
                fn = tc["function"]
                try:
                    inp = json.loads(fn["arguments"])
                except ValueError:
                    inp = {}
                blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": inp})
            stop_reason = {"tool_calls": "tool_use", "length": "max_tokens"}.get(finish, "end_turn")
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
            self._json(200, {
                "id": "msg_" + uuid.uuid4().hex[:24],
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": blocks,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": int(usage.get("prompt_tokens", 0)),
                    "output_tokens": int(usage.get("completion_tokens", 0)),
                    "cache_read_input_tokens": cached,
                },
            })

        def _complete_responses(self, req: dict, job: dict) -> None:
            text = {"reasoning": [], "content": []}
            calls: list = []

            def emit(kind, payload):
                (calls if kind == "tool" else text[kind]).append(payload)

            try:
                usage, finish = engine.run(job, emit, self._gone)
            except ClientGone:
                return
            except Exception:
                traceback.print_exc()
                self._error(500, "server error", "server_error")
                return
            rid = "resp_" + uuid.uuid4().hex[:24]
            output: list[dict[str, Any]] = []
            if text["reasoning"]:
                output.append({
                    "id": "rs_" + uuid.uuid4().hex[:24],
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "".join(text["reasoning"])}],
                })
            body = "".join(text["content"])
            if body or not calls:
                output.append({
                    "id": "msg_" + uuid.uuid4().hex[:24],
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": body, "annotations": []}],
                })
            for tc in calls:
                fn = tc["function"]
                output.append({
                    "id": "fc_" + uuid.uuid4().hex[:24],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tc["id"],
                    "name": fn["name"],
                    "arguments": fn["arguments"],
                })
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
            self._json(200, {
                "id": rid,
                "object": "response",
                "created_at": int(time.time()),
                "status": "incomplete" if finish == "length" else "completed",
                "model": model_id,
                "output": output,
                "output_text": body,
                "usage": {
                    "input_tokens": int(usage.get("prompt_tokens", 0)),
                    "output_tokens": int(usage.get("completion_tokens", 0)),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                    "input_tokens_details": {"cached_tokens": cached},
                },
            })

        def _start_sse(self):
            wlock, started, broken = threading.Lock(), threading.Event(), threading.Event()

            def write(raw: bytes) -> None:
                with wlock:
                    try:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(raw), raw))
                        self.wfile.flush()
                    except OSError:
                        broken.set()
                        raise ClientGone()

            def keepalive() -> None:  # SSE comments while a long prompt prefills, so clients don't time out
                while not started.wait(10.0):
                    try:
                        write(b": prefilling\n\n")
                    except ClientGone:
                        return

            self.send_response(200)
            for k, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
                         ("Transfer-Encoding", "chunked"), ("Connection", "close")):
                self.send_header(k, v)
            self.end_headers()
            return write, started, broken, threading.Thread(target=keepalive, daemon=True)

        def _stream(self, req: dict, job: dict) -> None:
            cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            n_tools = [0]
            write, started, broken, ka = self._start_sse()

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

            try:
                chunk({"role": "assistant", "content": ""})
                ka.start()
                usage, finish = engine.run(job, emit, lambda: broken.is_set() or self._gone())
                started.set()
                ka.join()
                chunk({}, finish)
                if include_usage:
                    chunk(None, usage=usage)
                write(b"data: [DONE]\n\n")
                write(b"")
            except ClientGone:
                log("client disconnected; generation stopped")
                self.close_connection = True
            except Exception:
                traceback.print_exc()
                try:
                    write(b"data: " + json.dumps({"error": {"message": "server error", "type": "server_error",
                                                            "code": 500}}).encode() + b"\n\n")
                    write(b"data: [DONE]\n\n")
                    write(b"")
                except ClientGone:
                    pass
            finally:
                started.set()
                if ka.ident is not None:
                    ka.join()

        def _stream_anthropic(self, req: dict, job: dict) -> None:
            mid = "msg_" + uuid.uuid4().hex[:24]
            write, started, broken, ka = self._start_sse()
            cur_kind: list[str | None] = [None]
            cur_idx = [-1]

            def send_ev(name: str, data: dict) -> None:
                raw = f"event: {name}\ndata: ".encode() + json.dumps(data, ensure_ascii=False).encode() + b"\n\n"
                write(raw)

            def close_block() -> None:
                if cur_kind[0] is not None:
                    send_ev("content_block_stop", {"type": "content_block_stop", "index": cur_idx[0]})
                    cur_kind[0] = None

            def emit(kind, payload) -> None:
                started.set()
                if kind == "tool":
                    close_block()
                    cur_idx[0] += 1
                    fn = payload["function"]
                    send_ev("content_block_start", {
                        "type": "content_block_start",
                        "index": cur_idx[0],
                        "content_block": {"type": "tool_use", "id": payload["id"], "name": fn["name"], "input": {}},
                    })
                    send_ev("content_block_delta", {
                        "type": "content_block_delta",
                        "index": cur_idx[0],
                        "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                    })
                    send_ev("content_block_stop", {"type": "content_block_stop", "index": cur_idx[0]})
                    return
                btype = "thinking" if kind == "reasoning" else "text"
                if cur_kind[0] != btype:
                    close_block()
                    cur_idx[0] += 1
                    cur_kind[0] = btype
                    blk = {"type": "thinking", "thinking": ""} if btype == "thinking" else {"type": "text", "text": ""}
                    send_ev("content_block_start", {"type": "content_block_start", "index": cur_idx[0], "content_block": blk})
                delta = {"type": "thinking_delta", "thinking": payload} if btype == "thinking" else {"type": "text_delta", "text": payload}
                send_ev("content_block_delta", {"type": "content_block_delta", "index": cur_idx[0], "delta": delta})

            try:
                send_ev("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": mid,
                        "type": "message",
                        "role": "assistant",
                        "model": model_id,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": len(job.get("ids") or ()), "output_tokens": 0},
                    },
                })
                ka.start()
                usage, finish = engine.run(job, emit, lambda: broken.is_set() or self._gone())
                started.set()
                ka.join()
                close_block()
                stop_reason = {"tool_calls": "tool_use", "length": "max_tokens"}.get(finish, "end_turn")
                cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
                send_ev("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {
                        "input_tokens": int(usage.get("prompt_tokens", 0)),
                        "output_tokens": int(usage.get("completion_tokens", 0)),
                        "cache_read_input_tokens": cached,
                    },
                })
                send_ev("message_stop", {"type": "message_stop"})
                write(b"")
            except ClientGone:
                log("client disconnected; generation stopped")
                self.close_connection = True
            except Exception:
                traceback.print_exc()
                try:
                    send_ev("error", {"type": "error", "error": {"type": "api_error", "message": "server error"}})
                    write(b"")
                except ClientGone:
                    pass
            finally:
                started.set()
                if ka.ident is not None:
                    ka.join()

        def _stream_responses(self, req: dict, job: dict) -> None:
            rid, created = "resp_" + uuid.uuid4().hex[:24], int(time.time())
            write, started, broken, ka = self._start_sse()
            output_items: list[dict[str, Any]] = []
            text_buf: list[str] = []
            reason_buf: list[str] = []
            open_item: list[dict[str, Any] | None] = [None]

            def send_ev(ev_type: str, data: dict) -> None:
                payload = {"type": ev_type, **data}
                raw = f"event: {ev_type}\ndata: ".encode() + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
                write(raw)

            def close_open() -> None:
                it = open_item[0]
                if it is None:
                    return
                idx = it["index"]
                if it["kind"] == "reasoning":
                    full = "".join(reason_buf)
                    item = {"id": it["id"], "type": "reasoning", "status": "completed",
                            "summary": [{"type": "summary_text", "text": full}]}
                    send_ev("response.reasoning_summary_text.done", {"item_id": it["id"], "output_index": idx,
                                                                     "summary_index": 0, "text": full})
                    send_ev("response.output_item.done", {"output_index": idx, "item": item})
                    output_items.append(item)
                elif it["kind"] == "content":
                    full = "".join(text_buf)
                    part = {"type": "output_text", "text": full, "annotations": []}
                    item = {"id": it["id"], "type": "message", "status": "completed", "role": "assistant", "content": [part]}
                    send_ev("response.output_text.done", {"item_id": it["id"], "output_index": idx, "content_index": 0, "text": full})
                    send_ev("response.content_part.done", {"item_id": it["id"], "output_index": idx, "content_index": 0, "part": part})
                    send_ev("response.output_item.done", {"output_index": idx, "item": item})
                    output_items.append(item)
                open_item[0] = None

            def emit(kind, payload) -> None:
                started.set()
                if kind == "tool":
                    close_open()
                    idx = len(output_items)
                    fn = payload["function"]
                    fc_id = "fc_" + uuid.uuid4().hex[:24]
                    item_start = {"id": fc_id, "type": "function_call", "status": "in_progress",
                                  "call_id": payload["id"], "name": fn["name"], "arguments": ""}
                    send_ev("response.output_item.added", {"output_index": idx, "item": item_start})
                    send_ev("response.function_call_arguments.delta",
                            {"item_id": fc_id, "output_index": idx, "delta": fn["arguments"]})
                    send_ev("response.function_call_arguments.done",
                            {"item_id": fc_id, "output_index": idx, "arguments": fn["arguments"]})
                    item_done = {**item_start, "status": "completed", "arguments": fn["arguments"]}
                    send_ev("response.output_item.done", {"output_index": idx, "item": item_done})
                    output_items.append(item_done)
                    return
                if open_item[0] is None or open_item[0]["kind"] != kind:
                    close_open()
                    idx = len(output_items)
                    if kind == "reasoning":
                        item_id = "rs_" + uuid.uuid4().hex[:24]
                        open_item[0] = {"kind": "reasoning", "id": item_id, "index": idx}
                        send_ev("response.output_item.added", {
                            "output_index": idx,
                            "item": {"id": item_id, "type": "reasoning", "status": "in_progress", "summary": []},
                        })
                    else:
                        item_id = "msg_" + uuid.uuid4().hex[:24]
                        open_item[0] = {"kind": "content", "id": item_id, "index": idx}
                        send_ev("response.output_item.added", {
                            "output_index": idx,
                            "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
                        })
                        send_ev("response.content_part.added", {
                            "item_id": item_id, "output_index": idx, "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        })
                it = open_item[0]
                if kind == "reasoning":
                    reason_buf.append(payload)
                    send_ev("response.reasoning_summary_text.delta",
                            {"item_id": it["id"], "output_index": it["index"], "summary_index": 0, "delta": payload})
                else:
                    text_buf.append(payload)
                    send_ev("response.output_text.delta",
                            {"item_id": it["id"], "output_index": it["index"], "content_index": 0, "delta": payload})

            base_resp = {"id": rid, "object": "response", "created_at": created, "status": "in_progress",
                         "model": model_id, "output": []}
            try:
                send_ev("response.created", {"response": base_resp})
                send_ev("response.in_progress", {"response": base_resp})
                ka.start()
                usage, finish = engine.run(job, emit, lambda: broken.is_set() or self._gone())
                started.set()
                ka.join()
                close_open()
                cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
                final_resp = {
                    **base_resp,
                    "status": "incomplete" if finish == "length" else "completed",
                    "output": output_items,
                    "output_text": "".join(text_buf),
                    "usage": {
                        "input_tokens": int(usage.get("prompt_tokens", 0)),
                        "output_tokens": int(usage.get("completion_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                        "input_tokens_details": {"cached_tokens": cached},
                    },
                }
                send_ev("response.completed", {"response": final_resp})
                write(b"")
            except ClientGone:
                log("client disconnected; generation stopped")
                self.close_connection = True
            except Exception:
                traceback.print_exc()
                try:
                    send_ev("error", {"message": "server error", "code": 500})
                    write(b"")
                except ClientGone:
                    pass
            finally:
                started.set()
                if ka.ident is not None:
                    ka.join()

    return Handler


def add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="packed Inkling-2-Mini directory or Hugging Face repo ID")
    p.add_argument("--device", default="mlx", help="mlx (Apple Silicon), cuda, or cuda:N")
    p.add_argument("--drafter", default=None, help="DSpark drafter directory, .safetensors file or Hugging Face repo ID")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1, network listening requires LAMQ_API_KEY)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model-id", default=None, help="model id reported by /v1/models (default: the model's name)")
    p.add_argument("--context", type=int, default=32768, help="prompt + completion token limit (default 32768)")
    p.add_argument("--max-tokens", type=int, default=8192, help="completion tokens when a request sets none")
    p.add_argument("--reasoning-effort", default="medium", choices=EFFORTS, help="default chat template reasoning effort")
    p.add_argument("--no-thinking", action="store_true",
                   help="CUDA only: force the none template setting and block the thinking-channel token")
    p.add_argument("--temp", type=float, default=0.0, help="temperature when a request sets none (0 = greedy)")
    p.add_argument("--top-p", type=float, default=1.0, help="top-p when a request sets none")
    p.add_argument("--prompt-cache", type=int, default=0, help="local single-user prompt-prefix snapshots kept (default 0)")
    p.add_argument("--prompt-cache-dir", default=None, help="persist prompt-prefix snapshots on disk across restarts (loopback only)")
    p.add_argument("--prompt-cache-gb", type=float, default=20.0, help="disk budget in GiB for --prompt-cache-dir (default 20)")
    p.add_argument("--no-repeat-guard", action="store_true", help="disable the stateless agent tool-loop repetition guard")
    p.add_argument("--prefill-chunk", type=int, default=-1, help="tokens per prefill chunk (-1 automatic, 0 one pass)")
    p.add_argument("--max-batch", type=int, default=32,
                   help="max concurrent batched Inkling MLX requests (default 32; 1 = legacy single-stream)")
    p.add_argument("--prefill-step", type=int, default=512,
                   help="tokens per interleaved prefill quantum in batched mode (default 512)")
    p.add_argument("--dspark-max-batch", type=int, default=4,
                   help="load-adaptive DSpark cutoff in batched mode: use DSpark up to B active requests (default 4)")
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="disable automatic zero-copy shared-prefix KV reuse in batched mode")
    p.add_argument("--no-verify", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--ui", action="store_true",
                   help="serve a browser chat page at / and accept that page's same-origin requests")


def serve(args: argparse.Namespace) -> int:
    api_key = os.environ.get("LAMQ_API_KEY")
    loopback = _is_loopback(args.host)
    if not loopback and not api_key:
        raise SystemExit("network listening requires LAMQ_API_KEY")
    max_batch = getattr(args, "max_batch", 32)
    prefill_step = getattr(args, "prefill_step", 512)
    dspark_max_batch = getattr(args, "dspark_max_batch", 4)
    if not (0 <= args.prompt_cache <= 8 and args.context >= 2 and args.max_tokens > 0 and args.prefill_chunk >= -1
            and 0 <= args.temp <= 2 and 0 <= args.top_p <= 1 and getattr(args, "prompt_cache_gb", 20.0) >= 0
            and 1 <= max_batch <= 128 and prefill_step >= 64 and dspark_max_batch >= 0):
        raise SystemExit("invalid server limits or sampling defaults")
    if args.device == "mlx":
        if getattr(args, "no_thinking", False):
            raise SystemExit("--no-thinking requires a CUDA device")
        engine_class = Engine
    elif args.device == "cuda" or (args.device.startswith("cuda:") and args.device[5:].isdecimal()):
        from runtime_cuda.server import Engine as CudaEngine

        if args.prefill_chunk != -1:
            raise SystemExit("--prefill-chunk is an MLX option")
        if args.top_p != 1.0:
            raise SystemExit("CUDA serving currently requires --top-p 1")
        engine_class = CudaEngine
    else:
        raise SystemExit("--device must be mlx, cuda, or cuda:N")
    if not loopback or api_key:
        args.prompt_cache = 0
        args.prompt_cache_dir = None
    model_id = args.model_id or str(args.model).rstrip("/").split("/")[-1]
    with closing(engine_class(args)) as engine:
        handler = make_handler(engine, model_id, api_key=api_key, loopback=loopback, ui=args.ui)
        max_conns = max(MAX_CONNECTIONS, int(getattr(engine, "max_concurrent", 1)) + 16)
        with BoundedHTTPServer((args.host, args.port), handler, max_connections=max_conns) as srv:
            if not args.no_warmup:
                engine.warmup()
            address = f"[{args.host}]" if ":" in args.host else args.host
            batch_note = f", max_batch {args.max_batch}" if getattr(engine, "scheduler", None) is not None else ""
            log(f"serving {model_id!r} on http://{address}:{srv.server_port}/v1 (context {args.context}, "
                f"reasoning_effort {args.reasoning_effort}, temp {args.temp}{batch_note})")
            if args.ui and loopback:
                log(f"chat page on http://{address}:{srv.server_port}/")
            elif args.ui:
                log("browser chat enabled at / on your HTTPS proxy")
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lamq serve", description="OpenAI-compatible server for packed Inkling-2-Mini models.")
    add_serve_args(p)
    return serve(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
