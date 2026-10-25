#!/usr/bin/env python3
"""READY gate: prompt-length sweep for degenerate output on a freshly booted engine.

One fresh request per prompt length: a filler prefix plus the same short arithmetic question, temperature 0,
thinking disabled, 16 tokens. The expected answer is computed here (a * b), so the gate cannot drift from the
question. A row fails when:
  - the content's final integer is not exactly the product ('17*23 = 391' passes; '4010', '1391' do not),
  - the reply stopped (finish_reason 'stop') with fewer completion tokens than the product's digits plus the EOS
    token (SHORT_STOP: e.g. '17' + EOS = 3 tokens where 391 needs 4); never exempted,
  - the reply ran into the 16-token cap without stopping (finish_reason 'length'), or ended with any finish_reason
    other than 'stop',
  - the text is degenerate (one token repeated 8+ times, or 'ductduct'),
  - a sampled logprob is not finite (or is vLLM's -9999 floor), or on an HTTP error.
KNOWN lists fills whose documented answer differs from the product on this exact build with fp8 KV. Only that
exact full answer (content == '401') is reported as a WARNING instead of a failure, never a substring. Fill 3
(76 prompt tokens) answers 401 in every fp8 sweep run on its own (it has also answered 391 under load). That entry
was itself added after a failure and is kept only by an explicit ruling; no entry is added because a run failed.

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
FILLS = [0, 3, 5, 12, 15, 25, 30, 40, 60, 75, 90, 150, 300, 600, 2400]   # 15 rows; 2400 fills is ~33K prompt tokens
KNOWN = {3: "401"}                                                       # exact full answers only (fp8 KV)
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
            "completion_tokens": d["usage"].get("completion_tokens"),
            "finish_reason": ch.get("finish_reason"),
            "content": m.get("content") or "",
            "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "logprobs": [t.get("logprob") for t in lp]}


def degenerate(text):
    return "ductduct" in text or re.search(r"(\S{2,12}?)\1{7,}", text) is not None


def final_int(content):
    """The answer's last integer, thousands commas removed; None if there is none."""
    m = re.findall(r"[-\u2212]?\d+(?:,\d{3})*", content)
    return m[-1].replace(",", "").replace("\u2212", "-") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="ornith-1.5-122b-exl3")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    product = str(A * B)
    needed = len(product) + 1                        # Qwen tokenizes digits one per token, plus the EOS token
    question = QUESTION.format(a=A, b=B)
    print(f"ENDPOINT={base} MODEL={a.model} QUESTION={A}*{B} EXPECT={product} "
          f"FILLS={len(FILLS)} KNOWN={KNOWN} SHORT_STOP_UNDER={needed}")

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
        ct = r["completion_tokens"]
        if r["finish_reason"] == "stop" and isinstance(ct, int) and ct < needed:
            hard.append(f"SHORT_STOP {ct} completion tokens < {needed}")
        if r["finish_reason"] == "length":
            hard.append(f"hit max_tokens {MAX_TOKENS} without stopping")
        elif r["finish_reason"] != "stop":
            hard.append(f"finish_reason {r['finish_reason']!r} is not 'stop'")
        text = r["reasoning"] + "|" + r["content"]
        note = ""
        if final_int(r["content"]) == product:
            pass
        elif KNOWN.get(n) is not None and r["content"] == KNOWN[n]:
            note = f"  [WARNING: KNOWN fill={n} answers exactly {KNOWN[n]}, not {product} (documented)]"
        else:
            hard.append(f"answer {final_int(r['content'])!r} is not {product}")
        if degenerate(text):
            hard.append("degenerate text run")
        fail = bool(hard)
        bad += fail
        snippet = text[:70].replace("\n", " ")
        print(f"fill={n:5d}  prompt_tokens={r['prompt_tokens']:6d} ctok={ct}  {'BAD ' if fail else 'ok  '}  "
              f"{snippet!r}{'  [' + ','.join(hard) + ']' if hard else ''}{note}", flush=True)

    print(f"NAN_SWEEP {len(FILLS) - bad}/{len(FILLS)} ok")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
