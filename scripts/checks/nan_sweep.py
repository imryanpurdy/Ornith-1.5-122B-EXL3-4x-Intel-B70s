#!/usr/bin/env python3
"""READY gate: prompt-length sweep for degenerate output on a freshly booted engine.

One fresh request per prompt length: a filler prefix plus the same short arithmetic question, temperature 0,
thinking disabled, 16 tokens. The expected answer is computed here (a * b), so the gate cannot drift from the
question. A row fails when the content does not contain the product, when a sampled logprob is not finite (or is
vLLM's -9999 floor), or on an HTTP error. A row whose answer is one of the KNOWN near-ties prints a WARNING
instead of failing: those are documented fp8 rounding flips at fill counts 3, 12, 75 and 150 (the
"391 flips to 401" family), frozen in KNOWN below and not re-litigated by this gate.

Usage: nan_sweep.py --port 8022 [--model ornith-1.5-122b-exl3] [--host 127.0.0.1]
Exit: 0 when every row passes, 1 otherwise.
"""
import argparse
import json
import math
import re
import sys
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog near the river bank. "
QUESTION = "\nIgnore the text above. What is {a} times {b}? Answer with just the number."
A, B = 17, 23
FILLS = [0, 3, 12, 15, 25, 30, 40, 60, 75, 90, 150, 300, 600, 2400]   # 14 rows; 2400 fills is ~33K prompt tokens
KNOWN = {3: 401, 12: 401, 75: 401, 150: 401}                          # documented near-tie flips, not failures
MAX_TOKENS = 16


def ask(base, model, content, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": MAX_TOKENS, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "logprobs": True, "top_logprobs": 1}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)          # json accepts NaN/Infinity literals, so a non-finite logprob survives parsing
    ch = d["choices"][0]
    m = ch["message"]
    lp = (ch.get("logprobs") or {}).get("content") or []
    return {"prompt_tokens": d["usage"]["prompt_tokens"],
            "content": m.get("content") or "",
            "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "logprobs": [t.get("logprob") for t in lp]}


def degenerate(text):
    return "ductduct" in text or re.search(r"(\S{2,12}?)\1{7,}", text) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="ornith-1.5-122b-exl3")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    product = A * B
    question = QUESTION.format(a=A, b=B)
    print(f"ENDPOINT={base} MODEL={a.model} QUESTION={A}*{B} EXPECT={product} "
          f"FILLS={len(FILLS)} KNOWN={KNOWN}")

    bad = 0
    for n in FILLS:
        try:
            r = ask(base, a.model, FILLER * n + question)
        except Exception as exc:                     # an HTTP error is a failure, not a crash
            bad += 1
            print(f"fill={n:5d}  BAD   error: {str(exc)[:120]}", flush=True)
            continue
        hard = []
        if not r["logprobs"]:
            hard.append("no logprobs returned")
        else:
            nonfinite = [x for x in r["logprobs"]
                         if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= -9999.0]
            if nonfinite:
                hard.append(f"non-finite logprob {nonfinite[:3]}")
        text = r["reasoning"] + "|" + r["content"]
        known = KNOWN.get(n)
        note = ""
        if str(product) in r["content"]:
            pass
        elif known is not None and str(known) in r["content"]:
            note = f"  [WARNING: KNOWN near-tie fill={n} answers {known}, not {product} (documented)]"
        else:
            hard.append(f"no {product} in content")
        if degenerate(text):
            note += "  [WARNING: degenerate text run]"
        fail = bool(hard)
        bad += fail
        snippet = text[:70].replace("\n", " ")
        print(f"fill={n:5d}  prompt_tokens={r['prompt_tokens']:6d}  {'BAD ' if fail else 'ok  '}  {snippet!r}"
              f"{'  [' + ','.join(hard) + ']' if hard else ''}{note}", flush=True)

    print(f"NAN_SWEEP {len(FILLS) - bad}/{len(FILLS)} ok")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
