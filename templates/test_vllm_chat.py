#!/usr/bin/env python3
"""CPU test of a server's client behaviour through vLLM 0.30's own serving path, with a scripted fake engine.

    python test_vllm_chat.py --model-dir <Ornith-1.5 checkpoint dir> -- serve <same dir> <the server's flags>

The serve args are parsed with vLLM's own `vllm serve` parser (make_arg_parser), the reasoning-parser plugin is
imported and the API args are validated as `vllm serve` does, the VllmConfig comes from
AsyncEngineArgs.from_cli_args(args).create_engine_config(), and OpenAIServingModels / OnlineRenderer /
OpenAIServingChat are built with the arguments init_app_state passes. Only the engine is fake (as in vLLM's own
tests/entrypoints/openai/chat_completion/test_serving_chat.py): it decodes the rendered prompt and answers like a
model that follows the prompt's think state:
    prompt ends '<think>\\n'               -> "<REASONING>\\n</think>\\n\\n<ANSWER>"
    prompt ends '<think>\\n\\n</think>\\n\\n' -> "<ANSWER>"
(ANSWER is a <tool_call> block when the request has tools.) Each case runs non-streaming and streaming, and the
response JSON is checked: thinking on -> reasoning == REASONING and content == "42"; thinking off -> no reasoning
and content == "42"; tool cases -> one get_weather(location=Paris) call. Also checked: the default sampling
parameters (the model's generation_config.json, no override), the SamplingParams each request gets, and that a
thinking budget set for an off request cannot fire (the budget state starts outside a think block).

Two more server configurations run for comparison (INFO, not pass/fail): the model's own template with the qwen3
parser (Ornith without this directory), and the template copy with the stock qwen3 parser.
If the quantization in the serve args is not registered (no EXL3 plugin on this machine), a stub config named
'exl3' is registered so ModelConfig can be built; it is never used to load weights.
Exit 0 iff every check of the configuration under test passes.
"""
import argparse
import asyncio
import json
import os
import sys
from unittest.mock import MagicMock

REASONING = "The user wants 17 + 25. That is 42."
ANSWER = "42"
TOOL_BLOCK = ("<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n"
              "</tool_call>")
ON_TAIL = "<|im_start|>assistant\n<think>\n"
OFF_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
PROD_OVERRIDE = '{"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}'
EXPECTED_SAMPLING = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}

# (label, request extras, thinking expected on?, tools?)
CASES = [
    ("default (thinking on)", {}, True, False),
    ("chat_template_kwargs enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}, False, False),
    ("reasoning_effort none", {"reasoning_effort": "none"}, False, False),
    ("reasoning_effort minimal", {"reasoning_effort": "minimal"}, False, False),
    ("chat_template_kwargs reasoning_effort off", {"chat_template_kwargs": {"reasoning_effort": "off"}}, False, False),
    ("chat_template_kwargs reasoning_effort none", {"chat_template_kwargs": {"reasoning_effort": "none"}}, False, False),
    ("chat_template_kwargs reasoning_effort minimal", {"chat_template_kwargs": {"reasoning_effort": "minimal"}},
     False, False),
    # the template opens the think block for any enable_thinking that is not False; the parser must agree
    ("chat_template_kwargs enable_thinking=0", {"chat_template_kwargs": {"enable_thinking": 0}}, True, False),
    ("chat_template_kwargs enable_thinking=''", {"chat_template_kwargs": {"enable_thinking": ""}}, True, False),
    ("reasoning_effort low", {"reasoning_effort": "low"}, True, False),
    ("reasoning_effort high", {"reasoning_effort": "high"}, True, False),
    ("tools, default (thinking on)", {}, True, True),
    ("tools, enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}, False, True),
    ("tools, reasoning_effort minimal", {"reasoning_effort": "minimal"}, False, True),
]
ALIAS_CASES = [  # OpenRouter-style {"reasoning": {"effort": ...}}: mapped only by the B70 patch (B70_REASONING_EFFORT_ALIAS=1)
    ("reasoning.effort none (B70 alias)", {"reasoning": {"effort": "none"}}, False, False),
    ("reasoning.effort minimal (B70 alias)", {"reasoning": {"effort": "minimal"}}, False, False),
]


def force_platform():
    """With no accelerator visible vLLM resolves an empty platform, and building the `vllm serve` parser fails
    (DeviceConfig cannot infer a device). Use the XPU platform class for argument parsing and config only; no
    device is ever opened (every probe of the device stays unused here)."""
    import vllm.platforms as platforms
    if platforms.current_platform.device_type:
        return platforms.current_platform.device_type
    from vllm.platforms.xpu import XPUPlatform
    platforms._current_platform = XPUPlatform()
    # Model Runner V2 requires Triton, which disables itself without a GPU driver; the runner choice is
    # engine-side only and does not touch the API layer under test.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    return "xpu (set by this test: no device visible; VLLM_USE_V2_MODEL_RUNNER=0 for config building)"


def ensure_quant_registered(name):
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    if name in QUANTIZATION_METHODS:
        return False

    @register_quantization_config(name)
    class _Stub(QuantizationConfig):  # never loads weights; only lets ModelConfig verify the checkpoint's method
        def get_name(self):
            return name

        @classmethod
        def get_supported_act_dtypes(cls):
            import torch
            return [torch.bfloat16, torch.float16]

        @classmethod
        def get_min_capability(cls):
            return 0

        @staticmethod
        def get_config_filenames():
            return []

        @classmethod
        def from_config(cls, config):
            return cls()

        def get_quant_method(self, layer, prefix):
            return None
    return True


def parse_serve_args(serve_args):
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.launchers.launcher import validate_api_server_args
    from vllm.reasoning import ReasoningParserManager
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    toks = list(serve_args)
    if toks and toks[0] == "serve":
        toks = toks[1:]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(toks)
    if getattr(args, "model_tag", None):
        args.model = args.model_tag
    validate_parsed_serve_args(args)
    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:   # setup_server, launcher.py
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)
    validate_api_server_args(args)
    return args


class Server:
    """init_app_state (entrypoints/launchers/api_server/app_state.py) + init_generate_state's OpenAIServingChat."""

    def __init__(self, serve_args):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.chat_utils import load_chat_template
        from vllm.entrypoints.launchers.cli_args import resolve_default_chat_template_kwargs
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.renderers.hf import HfRenderer
        from vllm.renderers.online_renderer import OnlineRenderer
        from vllm.tokenizers.registry import cached_tokenizer_from_config
        from vllm.v1.engine.async_llm import AsyncLLM

        self.args = args = parse_serve_args(serve_args)
        if args.quantization:
            self.stub_quant = ensure_quant_registered(args.quantization)
        # build_async_engine_client_from_engine_args: the whole VllmConfig, as `vllm serve` builds it
        self.vllm_config = vc = AsyncEngineArgs.from_cli_args(args).create_engine_config()
        self.model_config = mc = vc.model_config
        self.tokenizer = cached_tokenizer_from_config(mc)
        engine = MagicMock(spec=AsyncLLM)
        engine.errored = False
        engine.model_config = mc
        engine.vllm_config = vc
        engine.input_processor = MagicMock()
        engine.renderer = HfRenderer(vc, self.tokenizer)
        engine.generate = self.generate
        self.calls = []
        names = args.served_model_name or [args.model]
        models = OpenAIServingModels(engine_client=engine,
                                     base_model_paths=[BaseModelPath(name=n, model_path=args.model) for n in names])
        self.model_name = names[0]
        template = load_chat_template(args.chat_template)
        dctk = resolve_default_chat_template_kwargs(args)
        reasoning_parser = args.structured_outputs_config.reasoning_parser
        renderer = OnlineRenderer(
            model_config=mc, renderer=engine.renderer, request_logger=None, chat_template=template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
            enable_auto_tools=args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
            tool_parser=args.tool_call_parser, reasoning_parser=reasoning_parser, default_chat_template_kwargs=dctk)
        self.chat = OpenAIServingChat(
            engine, models, args.response_role, online_renderer=renderer, request_logger=None,
            chat_template=template, chat_template_content_format=args.chat_template_content_format,
            default_chat_template_kwargs=dctk, trust_request_chat_template=args.trust_request_chat_template,
            enable_auto_tools=args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
            tool_parser=args.tool_call_parser, reasoning_parser=reasoning_parser)
        self.reasoning_parser = reasoning_parser

    async def generate(self, engine_input, sampling_params, request_id, **kw):
        from vllm.outputs import CompletionOutput, RequestOutput
        from vllm.sampling_params import RequestOutputKind
        ids = list(engine_input["prompt_token_ids"])
        prompt = self.tokenizer.decode(ids, skip_special_tokens=False)
        answer = TOOL_BLOCK if "<tools>" in prompt else ANSWER
        if prompt.endswith(OFF_TAIL):
            text = answer
        elif prompt.endswith(ON_TAIL):
            text = REASONING + "\n</think>\n\n" + answer
        else:
            raise AssertionError(f"prompt ends in neither think state: {prompt[-80:]!r}")
        self.calls.append({"prompt": prompt, "prompt_ids": ids, "sp": sampling_params, "kw": kw, "output": text})
        out_ids = self.tokenizer.encode(text, add_special_tokens=False)

        def ro(t, tids, done):
            return RequestOutput(request_id=request_id, prompt=None, prompt_token_ids=ids, prompt_logprobs=None,
                                 outputs=[CompletionOutput(index=0, text=t, token_ids=tids, cumulative_logprob=None,
                                                           logprobs=None, finish_reason="stop" if done else None,
                                                           stop_reason=None)],
                                 finished=done)
        if sampling_params.output_kind == RequestOutputKind.DELTA:
            prev = ""
            for i in range(len(out_ids)):
                cur = self.tokenizer.decode(out_ids[:i + 1], skip_special_tokens=False)
                yield ro(cur[len(prev):], [out_ids[i]], i == len(out_ids) - 1)
                prev = cur
        else:
            yield ro(text, out_ids, True)

    async def request(self, body):
        """One chat completion. Returns (http_status, message dict with reasoning/content/tool_calls, call record)."""
        from pydantic import ValidationError
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
        try:
            req = ChatCompletionRequest.model_validate(body)    # FastAPI does this; a failure is a 400
        except ValidationError as e:
            return 400, {"error": str(e).splitlines()[0]}, None
        n0 = len(self.calls)
        res = await self.chat.create_chat_completion(req, None)
        call = self.calls[n0] if len(self.calls) > n0 else None
        if isinstance(res, ErrorResponse):
            return res.error.code, {"error": res.error.message}, call
        if not body.get("stream"):
            m = json.loads(res.model_dump_json())["choices"][0]["message"]
            return 200, {"reasoning": m.get("reasoning"), "content": m.get("content"),
                         "tool_calls": [(t["function"]["name"], json.loads(t["function"]["arguments"]))
                                        for t in m.get("tool_calls") or []]}, call
        acc = {"reasoning": "", "content": "", "tool_calls": {}}
        async for chunk in res:
            for line in chunk.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                for ch in d.get("choices", []):
                    delta = ch.get("delta") or {}
                    acc["reasoning"] += delta.get("reasoning") or ""
                    acc["content"] += delta.get("content") or ""
                    for tc in delta.get("tool_calls") or []:
                        slot = acc["tool_calls"].setdefault(tc.get("index", 0), {"name": "", "args": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
        tcs = [(s["name"], json.loads(s["args"]) if s["args"] else {}) for _, s in sorted(acc["tool_calls"].items())]
        call = self.calls[n0] if len(self.calls) > n0 else None      # a stream calls generate when iterated
        return 200, {"reasoning": acc["reasoning"] or None, "content": acc["content"] or None, "tool_calls": tcs}, call


def body_for(model, extras, tools, stream):
    b = {"model": model, "messages": [{"role": "user", "content": "What is 17 + 25? Answer with the number."}],
         "max_tokens": 256, "stream": stream}
    if tools:
        b["messages"] = [{"role": "user", "content": "What is the weather in Paris?"}]
        b["tools"] = TOOLS
        b["tool_choice"] = "auto"
    b.update(json.loads(json.dumps(extras)))
    return b


def verdict(msg, think_on, tools):
    """'ok' or what is wrong with a 200 response."""
    want_answer = [("get_weather", {"location": "Paris"})] if tools else None
    why = []
    if think_on:
        if (msg["reasoning"] or "").strip() != REASONING:
            why.append(f"reasoning {msg['reasoning']!r}")
    elif msg["reasoning"]:
        why.append(f"reasoning not empty: {msg['reasoning'][:40]!r}")
    if tools:
        if msg["tool_calls"] != want_answer:
            why.append(f"tool_calls {msg['tool_calls']}")
        if (msg["content"] or "").strip():
            why.append(f"content {msg['content'][:40]!r}")
    elif (msg["content"] or "").strip() != ANSWER:
        why.append(f"content {msg['content']!r}")
    return "ok" if not why else "; ".join(why)


def budget_state(server, call):
    """Thinking-budget state vLLM's sampler starts this request in (v1/sample/thinking_budget_state.py)."""
    import torch
    from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder
    budget = call["sp"].thinking_token_budget
    rc = server.vllm_config.reasoning_config
    if budget is None or rc is None or not rc.enabled:
        return budget, None
    holder = ThinkingBudgetStateHolder(rc, 4, 0, torch.device("cpu"), False)
    st = holder._init_state_entry(call["prompt_ids"], budget)
    return budget, st["in_think"]


async def run_config(label, serve_args, assert_it, report, expect_sampling=EXPECTED_SAMPLING):
    s = Server(serve_args)
    a = s.args
    print(f"\n== {label}\n   chat template {a.chat_template or '(model default)'}; reasoning parser "
          f"{s.reasoning_parser} (plugin {a.reasoning_parser_plugin or '-'}); tool parser {a.tool_call_parser}; "
          f"generation-config {s.model_config.generation_config}; override {s.model_config.override_generation_config}")
    print(f"   default sampling params (OpenAIServingChat.default_sampling_params): {s.chat.default_sampling_params}"
          + ("   [stub quant registered]" if getattr(s, "stub_quant", False) else ""))
    if assert_it:
        report(f"default sampling = {expect_sampling}",
               s.chat.default_sampling_params == expect_sampling, str(s.chat.default_sampling_params))
        report("no --override-generation-config", s.model_config.override_generation_config == {},
               str(s.model_config.override_generation_config))
    b70 = hasattr(__import__("vllm.entrypoints.openai.chat_completion.protocol", fromlist=["x"]),
                  "_b70_thinking_budget")
    alias_on = b70 and os.environ.get("B70_REASONING_EFFORT_ALIAS") == "1"
    cases = CASES + (ALIAS_CASES if alias_on else [])
    print(f"   {'case':47s} {'mode':6s} {'prompt tail':9s} {'reasoning':10s} {'content/tool':18s} "
          f"{'T/topk/topp/pres':20s} {'budget(in_think)':16s} verdict")
    for name, extras, think_on, tools in cases:
        for stream in (False, True):
            code, msg, call = await s.request(body_for(s.model_name, extras, tools, stream))
            if code != 200:
                v = f"HTTP {code}: {msg.get('error', '')[:80]}"
                print(f"   {name:47s} {'stream' if stream else 'full':6s} {v}")
                if assert_it:
                    report(f"{name} {'stream' if stream else 'full'}", False, v)
                continue
            v = verdict(msg, think_on, tools)
            sp = call["sp"]
            tail = "off" if call["prompt"].endswith(OFF_TAIL) else "on"
            budget, in_think = budget_state(s, call)
            shown = (msg["tool_calls"][0][0] if msg["tool_calls"] else repr((msg["content"] or "")[:14]))
            print(f"   {name:47s} {'stream' if stream else 'full':6s} {tail:9s} "
                  f"{'yes' if msg['reasoning'] else '-':10s} {shown:18s} "
                  f"{f'{sp.temperature:g}/{sp.top_k}/{sp.top_p:g}/{sp.presence_penalty:g}':20s} "
                  f"{f'{budget}({in_think})' if budget is not None else '-':16s} {v}")
            if assert_it:
                tag = f"{name} {'stream' if stream else 'full'}"
                report(tag + ": prompt think state", (tail == "on") == think_on, tail)
                report(tag + ": response", v == "ok", v)
                report(tag + ": request sampling = default sampling",
                       all(getattr(sp, k) == v for k, v in expect_sampling.items()), str(sp))
                if b70:
                    report(tag + ": presence_penalty 0 (B70_DEFAULT_PRESENCE_PENALTY)", sp.presence_penalty == 0.0,
                           str(sp.presence_penalty))
                if not think_on and budget is not None:
                    report(tag + ": thinking budget cannot fire (state starts outside a think block)",
                           in_think is False, f"budget {budget} in_think {in_think}")
                rk = call["kw"].get("reasoning_parser_kwargs") or {}
                report(tag + ": engine gets reasoning_ended = prompt closed the think block",
                       call["kw"].get("reasoning_ended") is (not think_on),
                       f"reasoning_ended={call['kw'].get('reasoning_ended')} ctk={rk.get('chat_template_kwargs')}")
    # 'off' is not a value of the top-level field in vLLM 0.30 (protocol.py: Literal[none, minimal, ...])
    code, msg, _ = await s.request(body_for(s.model_name, {"reasoning_effort": "off"}, False, False))
    print(f"   top-level reasoning_effort 'off' -> HTTP {code} ({msg.get('error', 'ok')[:90]})")
    if assert_it:
        report("top-level reasoning_effort 'off' is a 400 (schema), use chat_template_kwargs", code == 400, str(code))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="checkpoint dir; must be the model the serve args name")
    ap.add_argument("--expect-sampling", type=json.loads, default=EXPECTED_SAMPLING,
                    help="JSON dict the server must serve as default sampling (default: Ornith-1.5's 0.6/20/0.95)")
    ap.add_argument("serve_args", nargs=argparse.REMAINDER, help="-- serve <model> <flags...> (under test)")
    a = ap.parse_args()
    serve = [t for t in a.serve_args if t != "--"] if a.serve_args else []
    if not serve:
        ap.error("pass the serve args after --")
    plat = force_platform()
    import vllm
    print(f"vllm {vllm.__version__}; platform {plat}; serve args: {' '.join(serve)}")
    passed, failed = 0, []

    def report(name, ok, detail=""):
        nonlocal passed
        if ok:
            passed += 1
        else:
            failed.append(name)
            print(f"   FAIL {name}: {detail}")

    def without(args, *flags):
        out, skip = [], False
        for t in args:
            if skip:
                skip = False
                continue
            if t in flags:
                skip = True
                continue
            out.append(t)
        return out

    lane = serve      # the configuration under test
    model_tok = serve[serve.index("serve") + 1] if "serve" in serve[:1] and len(serve) > 1 else None
    report("serve args name --model-dir as the model",
           model_tok is not None and os.path.realpath(model_tok) == os.path.realpath(a.model_dir), str(model_tok))
    report("serve args name --chat-template", "--chat-template" in lane)
    report("serve args name --reasoning-parser-plugin", "--reasoning-parser-plugin" in lane)
    report("serve args do not override the generation config", "--override-generation-config" not in lane)
    stock = without(lane, "--chat-template", "--reasoning-parser", "--reasoning-parser-plugin") + \
        ["--reasoning-parser", "qwen3"]
    copy_qwen3 = without(lane, "--reasoning-parser", "--reasoning-parser-plugin") + ["--reasoning-parser", "qwen3"]
    prod_sampling = without(stock, "--generation-config") + ["--override-generation-config", PROD_OVERRIDE]

    async def go():
        await run_config("ASSERTED: the serve args under test", lane, True, report, a.expect_sampling)
        await run_config("INFO: model's own template + qwen3 parser (Ornith without templates/)", stock, False, report)
        await run_config("INFO: template copy + stock qwen3 parser (why the parser plugin exists)", copy_qwen3, False,
                         report)
        s = Server(prod_sampling)
        print(f"\n== INFO: Flash-Next prod's --override-generation-config on this checkpoint\n"
              f"   default sampling params: {s.chat.default_sampling_params}  (presence_penalty is not one of the "
              f"keys get_diff_sampling_param keeps)")
    asyncio.run(go())
    print(f"\nVLLM_CHAT_TEST {'PASS' if not failed else 'FAIL'} {passed} passed, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
