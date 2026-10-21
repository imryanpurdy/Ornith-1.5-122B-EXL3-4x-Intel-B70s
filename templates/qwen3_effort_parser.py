# SPDX-License-Identifier: MIT
"""qwen3_effort: vLLM's qwen3 reasoning parser, plus the reasoning_effort switch of ornith-chat-template.jinja.

Load it with
    --reasoning-parser-plugin <path>/qwen3_effort_parser.py --reasoning-parser qwen3_effort
together with
    --chat-template <path>/ornith-chat-template.jinja

Why it exists. vLLM 0.30's qwen3 parser (vllm/parser/qwen3.py, Qwen3Parser.__init__) decides from
chat_template_kwargs["enable_thinking"] alone (default True) whether the model output starts inside a think
block. The chat template copy also closes the think block for reasoning_effort none/off/minimal. For two of
those requests vLLM hands the parser enable_thinking=True or nothing:
  - top-level reasoning_effort "minimal": ChatCompletionRequest.build_chat_params sets
    enable_thinking = (reasoning_effort != "none") = True;
  - chat_template_kwargs {"reasoning_effort": "off" | "none" | "minimal"} with no top-level field: no
    enable_thinking at all.
A non-streaming response would then return the answer as reasoning and content None. (Streaming happens to
work, because it checks the prompt for a closing </think> first.)

What it changes. Only the thinking decision: thinking is off when enable_thinking is False (as before) or
when reasoning_effort is a string that is "none", "off" or "minimal" after strip() and lower() -- the same
test, on the same merged chat_template_kwargs, as the template. Everything else is Qwen3Parser unchanged
(same grammar, same tool-call handling, same token ids).
"""
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.qwen3 import Qwen3Parser
from vllm.reasoning import ReasoningParserManager

PARSER_NAME = "qwen3_effort"
OFF_EFFORTS = ("none", "off", "minimal")


def thinking_off(chat_template_kwargs) -> bool:
    """The template's test: enable_thinking is false, or reasoning_effort is none/off/minimal."""
    ctk = chat_template_kwargs if isinstance(chat_template_kwargs, dict) else {}
    if ctk.get("enable_thinking") is False:
        return True
    effort = ctk.get("reasoning_effort")
    return isinstance(effort, str) and effort.strip().lower() in OFF_EFFORTS


class Qwen3EffortParser(Qwen3Parser):
    def __init__(self, tokenizer, tools=None, **kwargs):
        ctk = kwargs.get("chat_template_kwargs") or {}
        if thinking_off(ctk) and ctk.get("enable_thinking") is not False:
            kwargs["chat_template_kwargs"] = {**ctk, "enable_thinking": False}
        super().__init__(tokenizer, tools, **kwargs)


Qwen3EffortReasoningAdapter, Qwen3EffortToolAdapter = make_adapters(Qwen3EffortParser)
ReasoningParserManager.register_module(PARSER_NAME, module=Qwen3EffortReasoningAdapter)
