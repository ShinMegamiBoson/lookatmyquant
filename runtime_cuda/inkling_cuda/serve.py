"""Translate Inkling token frames into OpenAI chat deltas."""
import json

END = "<|end_message|>"
MODEL = "<|message_model|>"
TOOL = "<|content_invoke_tool_json|>"
TEXT = "<|content_text|>"
THINK = "<|content_thinking|>"


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
