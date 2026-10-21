# Chat template and reasoning parser for Ornith-1.5

These files set what clients get when they turn thinking off. They are a pair: serve both, or neither.

| file | what it is |
|---|---|
| `ornith-chat-template.jinja` | Ornith-1.5's own `chat_template.jinja` with one condition widened (see below). |
| `qwen3_effort_parser.py` | vLLM reasoning-parser plugin `qwen3_effort`: vLLM's `qwen3` parser with the same thinking switch as the template. |
| `test_chat_template.py` | CPU test (transformers only): the copy against the model's own template. |
| `test_vllm_chat.py` | CPU test (vLLM 0.30, no GPU): a request's whole path through vLLM's serving code, with a fake engine. |

Serve flags (vLLM 0.30):

```
--chat-template <dir>/ornith-chat-template.jinja
--reasoning-parser qwen3_effort --reasoning-parser-plugin <dir>/qwen3_effort_parser.py
--enable-auto-tool-choice --tool-call-parser qwen3_xml
--generation-config auto          # sampling defaults from the model's generation_config.json; no override
```

## What changes for clients

Ornith's own template closes the think block (`<think>\n\n</think>\n\n` after `<|im_start|>assistant\n`) only
for `chat_template_kwargs: {"enable_thinking": false}`. This copy also closes it when `reasoning_effort` is
`none`, `off` or `minimal` (a string, compared after trim and lower-casing). Everything else renders
byte-identically: the test renders 7 conversations (system, multi-turn with reasoning, tools and tool results,
image parts) with and without a generation prompt under 30 sets of template arguments and compares them with the
source template.

| request | Ornith's template + `qwen3` | this template + `qwen3_effort` |
|---|---|---|
| no thinking control | thinks | thinks |
| `chat_template_kwargs: {"enable_thinking": false}` | no thinking | no thinking |
| `reasoning_effort: "none"` | no thinking (vLLM sets `enable_thinking=false` itself) | no thinking |
| `reasoning_effort: "minimal"` | thinks | no thinking |
| `chat_template_kwargs: {"reasoning_effort": "off" \| "none" \| "minimal"}` | thinks | no thinking |
| `reasoning_effort: "low"` ... `"ultra"` | thinks | thinks |
| `reasoning_effort: "off"` (top level) | HTTP 400 | HTTP 400 |

"No thinking" means the response has no `reasoning` field and the answer is in `content`, streaming or not.

## Why the parser is needed

vLLM 0.30's `qwen3` parser decides from `chat_template_kwargs["enable_thinking"]` alone (default true) whether the
output starts inside a think block (`vllm/parser/qwen3.py`, `Qwen3Parser.__init__`). vLLM sets that key from the
top-level `reasoning_effort` as `reasoning_effort != "none"` (`ChatCompletionRequest.build_chat_params`), so for
`minimal` it says true, and for a `reasoning_effort` inside `chat_template_kwargs` it says nothing. With the
template alone, those requests get a closed think block in the prompt, and a non-streaming response reports the
answer as `reasoning` with `content: null`. Streaming happens to be right, because it first checks the prompt for
a closing `</think>`. `qwen3_effort` applies the template's own test to the same arguments; nothing else differs
from `qwen3`.

## How `reasoning_effort` reaches the template in vLLM 0.30

- The request's top-level `reasoning_effort` must be one of `none, minimal, low, medium, high, xhigh, max, ultra`
  (`vllm/entrypoints/openai/chat_completion/protocol.py`). Anything else, including `off`, is a 400. Send `off`
  as `chat_template_kwargs.reasoning_effort` instead.
- `build_chat_params` passes it to the template as `reasoning_effort`. When the request also has a
  `chat_template_kwargs.reasoning_effort`, the top-level value wins; when the top-level field is unset, the
  `chat_template_kwargs` value is used.
- vLLM passes a template only the variables the template mentions (`resolve_chat_template_kwargs`). Ornith's own
  template never mentions `reasoning_effort`, so it never sees one; this copy does.
- A server-side thinking budget keyed by effort (some vLLM builds set one for `minimal`) cannot fire on these
  requests: the sampler starts them outside a think block because the prompt already closed it. `test_vllm_chat.py`
  checks this whenever the build sets a budget.

## Running the tests

Both run on CPU in the vLLM 0.30 image; no GPU is opened. `<model>` is an Ornith-1.5 checkpoint directory (its
`chat_template.jinja`, tokenizer and `generation_config.json` are read).

```
python test_chat_template.py --model-dir <model>
python test_vllm_chat.py --model-dir <model> -- serve <model> <the serve flags above and the rest of yours>
```

Without a GPU driver, `test_vllm_chat.py` builds vLLM's config with the XPU platform class and
`VLLM_USE_V2_MODEL_RUNNER=0` (engine-side settings only); if the EXL3 plugin is not installed it registers a stub
quantization name so the config can be built. It never loads weights.

## Origin and license

`ornith-chat-template.jinja` is derived from the `chat_template.jinja` of `ornith-ai/Ornith-1.5-122B` (MIT, as
declared on its model card), shipped byte-identical in `Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw`; its header
records the source's sha256. The parser plugin subclasses vLLM's (Apache-2.0) parser classes and copies no vLLM
code.
