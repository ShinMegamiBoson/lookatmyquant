import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel

from runtime_cuda.inkling_cuda.serve import END, MODEL, TEXT, THINK, TOOL, assistant_message
from runtime_cuda.server import Engine

TOOLS = [{"type": "function", "function": {"name": "propose_edit"}}]
MESSAGE = {"role": "user", "content": "A question"}


def endpoint(generated, *, no_thinking=False, prompt_tokens=2):
    values = [
        "[UNK]", MODEL, TEXT, TOOL, END, THINK, "propose_edit",
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
    args = SimpleNamespace(model="tiny", device="cuda", context=1024, drafter=None,
                           no_thinking=no_thinking, reasoning_effort="low", max_tokens=100, temp=0., top_p=1.)
    model = SimpleNamespace(context_limit=1024)
    with (patch("runtime_cuda.server.open_model", return_value=nullcontext((model, tokenizer))),
          patch("runtime_cuda.server.eos_ids", return_value={table["<eos>"]})):
        engine = Engine(args)
    return engine, iter(table[piece] for piece in generated)


def test_native_tool_call_requires_completed_declared_frame():
    for text in (
        MODEL + "propose_edit" + TOOL + '{"old":"a","new":"b"}',
        MODEL + "other" + TOOL + '{"old":"a","new":"b"}' + END,
    ):
        message, reason = assistant_message(text, TOOLS, "one")
        assert "tool_calls" not in message
        assert reason == "stop"


def test_plain_content_and_history_arguments():
    message, _ = assistant_message(MODEL + TEXT + "All done." + END, None, "one")
    assert message["content"] == "All done."
    engine, _ = endpoint([], no_thinking=True)
    call = {"function": {"name": "propose_edit", "arguments": '{"new":"b","old":"a"}'}}
    req = {"messages": [{"role": "assistant", "tool_calls": [call]}], "tools": TOOLS}
    engine.render(req)
    rendered = engine.tokenizer.apply_chat_template.call_args
    assert rendered.args[0][0]["tool_calls"][0]["function"]["arguments"] == {"old": "a", "new": "b"}
    assert rendered.kwargs["reasoning_effort"] == "none"
    assert call["function"]["arguments"] == '{"new":"b","old":"a"}'


def test_stream_emits_text_then_declared_tool():
    engine, output = endpoint([MODEL, TEXT, "I will edit the source.", END, MODEL, "propose_edit",
                               TOOL, '{"old":"a","new":"b"}', END], no_thinking=True)
    request = {"messages": [MESSAGE], "tools": TOOLS, "max_tokens": 12}
    events = []
    with patch("runtime_cuda.server.stream_tokens", return_value=(t for t in output)) as run:
        usage, finish = engine.run(engine.prepare(request), lambda *ev: events.append(ev), lambda: False)
    assert usage == {"prompt_tokens": 2, "completion_tokens": 9, "total_tokens": 11}
    assert finish == "tool_calls"
    assert run.call_args.kwargs["blocked_token"] == engine.thinking
    assert events[0] == ("content", "I will edit the source.")
    kind, call = events[1]
    assert kind == "tool" and call["function"]["name"] == "propose_edit"
    assert json.loads(call["function"]["arguments"]) == {"old": "a", "new": "b"}


def test_stream_separates_reasoning_and_text_and_stops_at_closed_answer():
    engine, output = endpoint([MODEL, THINK, "let me work it out", END, MODEL, TEXT, "391", END, " more"])
    events = []
    with patch("runtime_cuda.server.stream_tokens", return_value=(t for t in output)):
        usage, finish = engine.run(engine.prepare({"messages": [MESSAGE]}), lambda *ev: events.append(ev), lambda: False)
    assert events == [("reasoning", "let me work it out"), ("content", "391")]
    assert finish == "stop"
    assert usage["completion_tokens"] == 8


def test_explicit_thinking_choice_is_independent_of_prompt_length():
    engine, _ = endpoint([], prompt_tokens=640)
    request = {"messages": [MESSAGE]}
    assert engine.prepare(request)["blocked_token"] is None
    assert engine.prepare({**request, "reasoning_effort": "none"})["blocked_token"] == engine.thinking
    assert engine.prepare({**request, "reasoning_effort": "high"})["blocked_token"] is None
