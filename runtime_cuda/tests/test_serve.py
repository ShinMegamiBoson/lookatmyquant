import json
import unittest
from contextlib import nullcontext
from unittest.mock import Mock

from inkling_cuda.serve import Chat, END, MODEL, TEXT, TOOL, assistant_message, render_chat


class ChatFormatting(unittest.TestCase):
    tools = [{"type": "function", "function": {"name": "propose_edit"}}]

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
        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids.return_value = [2, 3, 4]
        tokenizer.encode.return_value = [10, 11]
        tokenizer.decode.return_value = (
            MODEL + TEXT + "I will edit the source." + END +
            MODEL + "propose_edit" + TOOL + '{"old":"a","new":"b"}' + END
        )
        session = Mock()
        session.predict.side_effect = [5, 6, 12, 2, 5, 13, 3, 14, 2]
        model = Mock()
        model.session.return_value = nullcontext(session)
        chat = Chat(tokenizer, model, {9}, 32, "inky", no_thinking=True)
        result = chat.complete({"messages": [], "tools": self.tools, "max_tokens": 12})
        self.assertEqual(result["usage"], {"prompt_tokens": 2, "completion_tokens": 9, "total_tokens": 11})
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(all(call.kwargs["blocked_token"] == 4 for call in session.predict.call_args_list))


if __name__ == "__main__":
    unittest.main()
