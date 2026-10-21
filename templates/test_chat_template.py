#!/usr/bin/env python3
"""CPU test of ornith-chat-template.jinja against the model's own chat_template.jinja (transformers only, no GPU).

    python test_chat_template.py --model-dir <Ornith-1.5 checkpoint dir> [--template <copy>]

1. Source identity: the copy minus its header comment, with the one changed condition put back, is
   byte-identical to the model's chat_template.jinja, and its sha256 is the one the header records.
2. Render equivalence with the tokenizer's apply_chat_template, over a matrix of conversations (plain, system,
   multi-turn with inline and field reasoning, tools + tool calls + tool results, image part, list-form system
   content) x add_generation_prompt x template kwargs:
     - every input without an off-effort renders byte-identically to the source;
     - every off-effort input (reasoning_effort none/off/minimal, any case/whitespace) renders exactly as the
       source does with enable_thinking=false.
3. Exact prompt tails for the request kinds a server sees (vLLM argument combinations).
Exit 0 iff every check passes.
"""
import argparse
import hashlib
import itertools
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_COND = "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
OFF_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
ON_TAIL = "<|im_start|>assistant\n<think>\n"

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]

CONVERSATIONS = {
    "user": ([{"role": "user", "content": "What is 17 + 25?"}], None),
    "system+user": ([{"role": "system", "content": "You are terse."},
                     {"role": "user", "content": "What is 17 + 25?"}], None),
    "multi-turn inline think": ([{"role": "user", "content": "What is 17 + 25?"},
                                 {"role": "assistant", "content": "<think>\nAdd them.\n</think>\n\n42"},
                                 {"role": "user", "content": "And 42 + 1?"}], None),
    "multi-turn reasoning field": ([{"role": "user", "content": "What is 17 + 25?"},
                                    {"role": "assistant", "reasoning_content": "Add them.", "content": "42"},
                                    {"role": "user", "content": "And 42 + 1?"}], None),
    "tools": ([{"role": "system", "content": "Use tools when useful."},
               {"role": "user", "content": "Weather in Paris?"},
               {"role": "assistant", "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "get_weather",
                                                                 "arguments": {"location": "Paris"}}}]},
               {"role": "tool", "content": "18 C, clear"},
               {"role": "user", "content": "Thanks. Short answer?"}], TOOLS),
    "image part": ([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe it."}]}],
                   None),
    "list system": ([{"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
                     {"role": "user", "content": "Hi"}], None),
}

ON_KWARGS = [{}, {"enable_thinking": True}, {"enable_thinking": "false"}, {"enable_thinking": None},
             *({"reasoning_effort": e} for e in ("low", "medium", "high", "xhigh", "max", "ultra", "default",
                                                  "", "nonee", "minimal-ish", None, 0, False, ["none"])),
             {"enable_thinking": True, "reasoning_effort": "high"}]
OFF_KWARGS = [{"enable_thinking": False},
              *({"reasoning_effort": e} for e in ("none", "off", "minimal", "None", " OFF ", "Minimal\n")),
              {"enable_thinking": True, "reasoning_effort": "minimal"},   # what vLLM sends for top-level minimal
              {"enable_thinking": False, "reasoning_effort": "none"},     # what vLLM sends for top-level none
              {"enable_thinking": False, "reasoning_effort": "high"},     # explicit false wins, as in the source
              {"enable_thinking": True, "reasoning_effort": "off"}]


def strip_header(text):
    m = re.match(r"\{#-.*?-#\}\n", text, re.S)
    if not m:
        raise AssertionError("copy has no leading {#- ... -#} header")
    return text[m.end():], text[:m.end()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--template", default=str(HERE / "ornith-chat-template.jinja"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import transformers

    src = (Path(a.model_dir) / "chat_template.jinja").read_text()
    copy = Path(a.template).read_text()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    passed, failed = 0, []

    def check(name, ok, detail=""):
        nonlocal passed
        if ok:
            passed += 1
        else:
            failed.append(name)
            print(f"FAIL {name} {detail}")

    print(f"transformers {transformers.__version__}; source {Path(a.model_dir) / 'chat_template.jinja'}; copy {a.template}")
    # 1. source identity
    body, header = strip_header(copy)
    new_lines = [ln for ln in body.splitlines(keepends=True) if "reasoning_effort" in ln]
    check("exactly one changed line", len(new_lines) == 1, f"{len(new_lines)} lines mention reasoning_effort")
    restored = body.replace(new_lines[0], OLD_COND) if new_lines else body
    check("copy minus header, condition restored == source (bytes)", restored == src)
    sha = hashlib.sha256(src.encode()).hexdigest()
    check("header records the source sha256", sha in header, sha)
    check("tokenizer serves the source template by default", tok.get_chat_template() == src)
    print(f"source sha256 {sha}; changed line:\n  {new_lines[0].strip() if new_lines else '-'}")

    def render(tpl, conv, tools, agp, kw):
        return tok.apply_chat_template(conv, tools=tools, chat_template=tpl, tokenize=False,
                                       add_generation_prompt=agp, **kw)

    # 2. render equivalence
    n_on = n_off = 0
    for (cname, (conv, tools)), agp in itertools.product(CONVERSATIONS.items(), (True, False)):
        for kw in ON_KWARGS:
            o, c = render(src, conv, tools, agp, kw), render(copy, conv, tools, agp, kw)
            check(f"on  {cname} agp={agp} {kw}", c == o)
            if agp:
                check(f"on tail {cname} {kw}", c.endswith(ON_TAIL), repr(c[-60:]))
            n_on += 1
        for kw in OFF_KWARGS:
            c = render(copy, conv, tools, agp, kw)
            o = render(src, conv, tools, agp, {**kw, "enable_thinking": False}) if agp else render(src, conv, tools, agp, kw)
            check(f"off {cname} agp={agp} {kw}", c == o)
            if agp:
                check(f"off tail {cname} {kw}", c.endswith(OFF_TAIL), repr(c[-60:]))
            n_off += 1
    print(f"render matrix: {len(CONVERSATIONS)} conversations x add_generation_prompt {{True, False}} x "
          f"{len(ON_KWARGS)} on-kwargs ({n_on} renders) + {len(OFF_KWARGS)} off-kwargs ({n_off} renders)")

    # 3. exact tails per request kind (single user turn; kwargs as vLLM 0.30 builds them)
    conv = CONVERSATIONS["user"][0]
    head = "<|im_start|>user\nWhat is 17 + 25?<|im_end|>\n"
    for label, kw, tail in (("default", {}, ON_TAIL),
                            ("enable_thinking false", {"enable_thinking": False}, OFF_TAIL),
                            ("reasoning_effort none", {"reasoning_effort": "none", "enable_thinking": False}, OFF_TAIL),
                            ("reasoning_effort minimal", {"reasoning_effort": "minimal", "enable_thinking": True}, OFF_TAIL),
                            ("chat_template_kwargs reasoning_effort off", {"reasoning_effort": "off"}, OFF_TAIL),
                            ("reasoning_effort high", {"reasoning_effort": "high", "enable_thinking": True}, ON_TAIL)):
        out = render(copy, conv, None, True, kw)
        check(f"exact prompt: {label}", out == head + tail, repr(out))
        print(f"  {label:44s} -> {out[len(head):]!r}")

    print(f"TEMPLATE_TEST {'PASS' if not failed else 'FAIL'} {passed} passed, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
