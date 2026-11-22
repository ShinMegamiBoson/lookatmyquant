import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from runtime_cuda.server import Engine as CudaEngine
from runtime_mlx.server import SPECIAL, Engine as MlxEngine, RequestError

END, MODEL, TEXT, THINK, TOOL, EOS = (SPECIAL[k] for k in ("end", "model", "text", "thinking", "tool_json", "eos"))
TOOLS = [{"type": "function", "function": {"name": "propose_edit"}}]
MESSAGE = {"role": "user", "content": "A question"}
ARGS = '{"old":"a","new":"b"}'


def adapters(generated, *, no_thinking=False, prompt_tokens=2):
    values = list(dict.fromkeys([*SPECIAL.values(), *generated, "<default_eos>"]))
    table = {piece: i for i, piece in enumerate(values)}
    ids = {key: table[special] for key, special in SPECIAL.items()}
    tokenizer = Mock()
    tokenizer.convert_tokens_to_ids.side_effect = table.__getitem__
    tokenizer.decode.side_effect = lambda tokens, **_: "".join(values[token] for token in tokens)
    tokenizer.encode.return_value = [0] * prompt_tokens
    tokenizer.apply_chat_template.side_effect = lambda _, *, tokenize, **kw: [0] * prompt_tokens if tokenize else "prompt"
    args = SimpleNamespace(model="tiny", device="cuda", context=1024, drafter=None, prefill_chunk=0,
                           no_thinking=no_thinking, reasoning_effort="low", max_tokens=100, temp=0., top_p=1.)
    model = SimpleNamespace(context_limit=1024, lm_head=SimpleNamespace(order_mx=SimpleNamespace(size=len(table))))
    with (patch("runtime_cuda.server.open_model", return_value=nullcontext((model, tokenizer))),
          patch("runtime_cuda.server.eos_ids", return_value={table["<default_eos>"]})):
        cuda = CudaEngine(args)
    mlx = MlxEngine.__new__(MlxEngine)
    mlx.args, mlx.tok, mlx.hf, mlx.ids, mlx.m = args, tokenizer, tokenizer, ids, model
    mlx.drafter = mlx.prompt_cache = None
    return cuda, mlx, [table[piece] for piece in generated]


def run_cuda(engine, request, output):
    events = []
    with patch("runtime_cuda.server.stream_tokens", return_value=(t for t in output)) as backend:
        usage, finish = engine.run(engine.prepare(request), lambda *ev: events.append(ev), lambda: False)
    return events, usage, finish, backend.call_args.kwargs


def run_mlx(engine, request, output):
    events = []
    mlx, mx, inkling = ModuleType("mlx"), ModuleType("mlx.core"), ModuleType("runtime_mlx.inkling")
    mlx.core = mx
    mx.reset_peak_memory = mx.clear_cache = lambda: None
    mx.get_peak_memory = lambda: 0
    inkling.stream_serial = lambda *args, **kwargs: (output for _ in range(1))
    inkling.stream_dspark = inkling.stream_serial
    with (patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx, "runtime_mlx.inkling": inkling}),
          patch("runtime_mlx.server.log")):
        usage, finish = engine._run(engine.prepare(request), lambda *ev: events.append(ev), lambda: False)
    return events, usage, finish


def assert_same_response(cuda, mlx, request, ids):
    events, usage, finish, backend = run_cuda(cuda, request, ids)
    mlx_events, mlx_usage, mlx_finish = run_mlx(mlx, request, ids)
    def comparable(events):
        return [(kind, data["function"] if kind == "tool" else data) for kind, data in events]

    assert (comparable(events), usage, finish) == (comparable(mlx_events), mlx_usage, mlx_finish)
    return events, usage, finish, backend


def test_tool_calls_require_completed_declared_frame():
    plain = [MODEL, TEXT, "Continuing normally.", END, EOS]
    for tool_tokens, tools, tail in (
        ([MODEL, "propose_edit", TOOL, ARGS], TOOLS, [EOS]),
        ([MODEL, "other", TOOL, ARGS, END], TOOLS, plain),
        ([MODEL, "propose_edit", TOOL, ARGS, END], [], plain),
        ([MODEL, "propose_edit", TOOL, '{"name":"propose_edit","args":{}', END], TOOLS, plain),
        ([MODEL, "propose_edit", TOOL, '{"name":"other","args":{}}', END], TOOLS, plain),
        ([MODEL, "propose_edit", TOOL, '{"arguments":[]}', END], TOOLS, plain),
        ([MODEL, "propose_edit", SPECIAL["tool_text"], ARGS, END], TOOLS, plain),
    ):
        cuda, mlx, ids = adapters(tool_tokens + tail)
        events, _, finish, _ = assert_same_response(cuda, mlx, {"messages": [MESSAGE], "tools": tools}, ids)
        assert events == ([("content", "Continuing normally.")] if tail == plain else [])
        assert finish == "stop"


def test_incoming_tools_and_history_arguments():
    cuda, mlx, _ = adapters([], no_thinking=True)
    call = {"function": {"name": "propose_edit", "arguments": '{"new":"b","old":"a"}'}}
    req = {"messages": [{"role": "assistant", "tool_calls": [call]}], "tools": TOOLS}
    for engine in (cuda, mlx):
        engine.render(req)
        rendered = engine.tok.apply_chat_template.call_args if engine is mlx else engine.tokenizer.apply_chat_template.call_args
        assert rendered.args[0][0]["tool_calls"][0]["function"]["arguments"] == {"old": "a", "new": "b"}
        assert rendered.kwargs["tools"] == TOOLS
        assert call["function"]["arguments"] == '{"new":"b","old":"a"}'
        for bad in ([{}], [{"type": "function", "function": {"name": ""}}]):
            try:
                engine.prepare(dict(req, tools=bad))
            except RequestError as e:
                assert e.status == 400
            else:
                raise AssertionError("invalid tool declaration was accepted")


def test_stream_emits_text_then_declared_tool():
    for header, body in (
        (["propose_edit"], ARGS),
        (["propose_edit"], '{"arguments":' + ARGS + '}'),
        (["propose_edit"], '{"name":"propose_edit","args":' + ARGS + '}'),
        ([], '{"name":"propose_edit","arguments":' + ARGS + '}'),
    ):
        generated = [MODEL, TEXT, "I will edit the source.", END, MODEL, *header, TOOL, body, END]
        cuda, mlx, ids = adapters(generated, no_thinking=True)
        request = {"messages": [MESSAGE], "tools": TOOLS, "max_tokens": 12}
        events, usage, finish, backend = assert_same_response(cuda, mlx, request, ids)
        assert usage["completion_tokens"] == len(generated)
        assert finish == "tool_calls"
        assert backend["blocked_token"] == cuda.ids["thinking"]
        assert events[0] == ("content", "I will edit the source.")
        kind, call = events[1]
        assert kind == "tool" and call["function"]["name"] == "propose_edit"
        assert json.loads(call["function"]["arguments"]) == {"old": "a", "new": "b"}


def test_stream_separates_reasoning_and_text_until_end_of_turn():
    generated = [MODEL, THINK, "let me work it out", END, MODEL, TEXT, "391", END, MODEL, TEXT, "Confirmed.", END]
    cuda, mlx, ids = adapters(generated + [EOS, "ignored"])
    events, usage, finish, _ = assert_same_response(cuda, mlx, {"messages": [MESSAGE]}, ids)
    assert events == [("reasoning", "let me work it out"), ("content", "391"), ("content", "\n\nConfirmed.")]
    assert finish == "stop"
    assert usage["completion_tokens"] == len(generated) + 1

    cuda, _, ids = adapters([MODEL, TEXT, "391", "<default_eos>"])
    events, _, finish, _ = run_cuda(cuda, {"messages": [MESSAGE]}, ids)
    assert events == [("content", "391")]
    assert finish == "stop"


def test_default_and_request_reasoning_are_template_only():
    for prompt_tokens in (2, 640):
        cuda, _, _ = adapters([], prompt_tokens=prompt_tokens)
        request = {"messages": [MESSAGE]}
        for settings, expected in (
            ({}, "low"),
            ({"reasoning_effort": "none"}, "none"),
            ({"reasoning_effort": 0}, 0),
            ({"reasoning_effort": "high"}, "high"),
        ):
            assert cuda.prepare({**request, **settings})["blocked_token"] is None
            assert cuda.tokenizer.apply_chat_template.call_args.kwargs["reasoning_effort"] == expected

    cuda.args.reasoning_effort = "none"
    assert cuda.prepare(request)["blocked_token"] is None
    assert cuda.tokenizer.apply_chat_template.call_args.kwargs["reasoning_effort"] == "none"


def test_explicit_cuda_serve_switch_preserves_hard_no_thinking():
    from runtime_cuda.cli import main as cuda_main
    from runtime_mlx.cli import main as lamq_main

    cuda, _, _ = adapters([])
    request = {"messages": [MESSAGE], "chat_template_kwargs": {"enable_thinking": False}}
    assert cuda.prepare(request)["blocked_token"] == cuda.ids["thinking"]
    assert cuda.tokenizer.apply_chat_template.call_args.kwargs["reasoning_effort"] == "none"

    cuda, _, _ = adapters([], no_thinking=True)
    assert cuda.prepare({"messages": [MESSAGE], "reasoning_effort": "high"})["blocked_token"] == cuda.ids["thinking"]
    assert cuda.tokenizer.apply_chat_template.call_args.kwargs["reasoning_effort"] == "none"
    with patch("runtime_mlx.server.serve", return_value=0) as run:
        assert cuda_main(["serve", "tiny", "--no-thinking"]) == 0
        assert run.call_args.args[0].device == "cuda"
        assert run.call_args.args[0].no_thinking is True
    try:
        lamq_main(["serve", "tiny", "--no-thinking"])
    except SystemExit as e:
        assert str(e) == "--no-thinking requires a CUDA device"
    else:
        raise AssertionError("MLX unexpectedly accepted the CUDA-only option")


if __name__ == "__main__":
    test_tool_calls_require_completed_declared_frame()
    test_incoming_tools_and_history_arguments()
    test_stream_emits_text_then_declared_tool()
    test_stream_separates_reasoning_and_text_until_end_of_turn()
    test_default_and_request_reasoning_are_template_only()
    test_explicit_cuda_serve_switch_preserves_hard_no_thinking()
    print("CUDA and MLX protocol tests passed")
