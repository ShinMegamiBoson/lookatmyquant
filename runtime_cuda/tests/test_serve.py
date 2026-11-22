import json
import unittest
from contextlib import nullcontext
from unittest.mock import Mock

from inkling_cuda.serve import Chat, END, MODEL, TEXT, TOOL, assistant_message, render_chat


class ChatFormatting(unittest.TestCase):
    tools = [{"type": "function", "function": {"name": "propose_edit"}}]

    def endpoint(self, generated, *, no_thinking=False, prompt_tokens=2):
        from tokenizers import Tokenizer
        from tokenizers.decoders import Fuse
        from tokenizers.models import WordLevel

        values = [
            "[UNK]", MODEL, TEXT, TOOL, END, "<|content_thinking|>", "propose_edit",
            "I will edit the source.", '{"old":"a","new":"b"}', "let me work it out",
            "391", "<eos>", " more",
        ]
        table = {piece: i for i, piece in enumerate(values)}
        backend = Tokenizer(WordLevel(table, unk_token="[UNK]"))
        backend.decoder = Fuse()
        tokenizer = Mock(backend_tokenizer=backend)
        tokenizer.convert_tokens_to_ids.side_effect = lambda pieces: [table[p] for p in pieces]
        tokenizer.decode.side_effect = backend.decode
        tokenizer.encode.return_value = [0] * prompt_tokens
        session = Mock()
        session.predict.side_effect = [table[piece] for piece in generated]
        model = Mock()
        model.session.return_value = nullcontext(session)
        chat = Chat(tokenizer, model, {table["<eos>"]}, 1024, "inky", no_thinking)
        return chat, session

    def test_native_call_is_structured_for_goose(self):
        text = MODEL + "propose_edit" + TOOL + '{"args":{"old":"a","new":"b"},"name":"propose_edit"}' + END
        message, reason = assistant_message(text, self.tools, "one")
        self.assertEqual(reason, "tool_calls")
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["id"], "call_one")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "propose_edit")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"old": "a", "new": "b"})

    def test_incomplete_and_undeclared_calls_stay_unexecutable(self):
        for text in (
            MODEL + "propose_edit" + TOOL + '{"old":"a","new":"b"}',
            MODEL + "other" + TOOL + '{"old":"a","new":"b"}' + END,
        ):
            message, reason = assistant_message(text, self.tools, "one")
            self.assertNotIn("tool_calls", message)
            self.assertEqual(reason, "stop")

    def test_plain_content_and_history_arguments(self):
        message, _ = assistant_message(MODEL + TEXT + "All done." + END, None, "one")
        self.assertEqual(message["content"], "All done.")
        tokenizer = Mock()
        call = {"function": {"name": "propose_edit", "arguments": '{"new":"b","old":"a"}'}}
        request = {"messages": [{"role": "assistant", "tool_calls": [call]}], "tools": self.tools}
        render_chat(tokenizer, request, no_thinking=True)
        rendered = tokenizer.apply_chat_template.call_args
        self.assertEqual(rendered.args[0][0]["tool_calls"][0]["function"]["arguments"], {"old": "a", "new": "b"})
        self.assertEqual(rendered.kwargs["reasoning_effort"], "none")
        self.assertEqual(call["function"]["arguments"], '{"new":"b","old":"a"}')

    def test_text_frame_can_precede_tool_in_same_completion(self):
        chat, session = self.endpoint(
            [MODEL, TEXT, "I will edit the source.", END, MODEL, "propose_edit", TOOL,
             '{"old":"a","new":"b"}', END],
            no_thinking=True,
        )
        result = chat.complete({"messages": [], "tools": self.tools, "max_tokens": 12})
        self.assertEqual(result["usage"], {"prompt_tokens": 2, "completion_tokens": 9, "total_tokens": 11})
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(all(call.kwargs["blocked_token"] == chat.thinking for call in session.predict.call_args_list))

    def test_stream_separates_reasoning_and_text_and_stops_at_closed_answer(self):
        chat, session = self.endpoint(
            [MODEL, "<|content_thinking|>", "let me work it out", END, MODEL, TEXT, "391", END, " more"],
        )
        request = {"messages": [{"role": "user", "content": "A question"}], "stream": True}
        chunks = list(chat.chunks(chat.prepare(request)))
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        self.assertEqual(deltas, [{"role": "assistant"}, {"reasoning_content": "let me work it out"},
                                  {"content": "391"}, {}])
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 8)
        self.assertEqual(session.predict.call_count, 8)

    def test_automatic_long_prompt_masks_thinking_but_explicit_effort_is_honored(self):
        chat, _ = self.endpoint([], prompt_tokens=640)
        request = {"messages": [{"role": "user", "content": "A long document"}]}
        prepared = chat.prepare(request)
        self.assertEqual(prepared.blocked_token, chat.thinking)
        self.assertEqual(chat.tokenizer.apply_chat_template.call_args.kwargs["reasoning_effort"], "none")
        self.assertIsNone(chat.prepare({**request, "reasoning_effort": "low"}).blocked_token)

    def test_tool_requests_keep_low_reasoning_at_long_prompt_lengths(self):
        chat, _ = self.endpoint([], prompt_tokens=640)
        request = {"messages": [{"role": "user", "content": "Use the tool"}], "tools": self.tools}
        self.assertIsNone(chat.prepare(request).blocked_token)
        self.assertEqual(chat.tokenizer.apply_chat_template.call_args.kwargs["reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()
