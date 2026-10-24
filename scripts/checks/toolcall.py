#!/usr/bin/env python3
"""READY gate: tool calling on a freshly booted engine (qwen3_xml parser + auto tool choice).

Three parts, all must pass:
  1. N sequential single-tool calls: finish_reason == "tool_calls", exactly one call, name == get_weather, and the
     arguments decode to {"location": <string containing "paris">}.
  2. A two-tool pick: the model must choose get_time (the question asks for the time), not get_weather.
  3. A nested-arguments case: create_calendar_event with event.title, a 2-item attendees list and
     event.when.{date,time} carrying the asked values.

Usage: toolcall.py --port 8022 [--model ornith-1.5-122b-exl3] [--count 3] [--host 127.0.0.1]
Exit: 0 when all three parts pass, 1 otherwise.
"""
import argparse
import json
import sys
import time
import urllib.request

TOOL_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather conditions for a city.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                       "required": ["location"]},
    },
}
TOOL_TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time for a city.",
        "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []},
    },
}
TOOL_NESTED = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "event": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "attendees": {"type": "array", "items": {"type": "string"}},
                        "when": {"type": "object",
                                 "properties": {"date": {"type": "string"}, "time": {"type": "string"}},
                                 "required": ["date", "time"]},
                    },
                    "required": ["title", "attendees", "when"],
                }
            },
            "required": ["event"],
        },
    },
}


def call(base, body, timeout=180):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8", errors="replace"))
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="ornith-1.5-122b-exl3")
    ap.add_argument("--count", type=int, default=3)
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    print(f"ENDPOINT={base} MODEL={a.model} SINGLE_CALLS={a.count}")
    fails = []

    for i in range(1, a.count + 1):
        body = {"model": a.model,
                "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
                "tools": [TOOL_WEATHER], "tool_choice": "auto", "max_tokens": 96, "temperature": 0.0}
        try:
            out, wall = call(base, body)
            ch = out["choices"][0]
            tcs = ch["message"].get("tool_calls") or []
            ok = ch.get("finish_reason") == "tool_calls" and len(tcs) == 1
            if ok:
                fn = tcs[0]["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = None
                ok = (fn.get("name") == "get_weather" and isinstance(args, dict)
                      and isinstance(args.get("location"), str) and "paris" in args["location"].lower())
            print(f"TC{i:02d} {'ok  ' if ok else 'BAD '} finish={ch.get('finish_reason')} wall={wall:.1f}s")
            if not ok:
                fails.append(f"tc{i}")
        except Exception as exc:
            print(f"TC{i:02d} BAD  error={str(exc)[:120]}")
            fails.append(f"tc{i}")
    print(f"TOOLCALL_VERDICT={a.count - len(fails)}/{a.count} fails={fails}")

    # multi-tool pick
    body = {"model": a.model,
            "messages": [{"role": "user", "content": "What time is it in Paris right now? Use the appropriate tool."}],
            "tools": [TOOL_WEATHER, TOOL_TIME], "tool_choice": "auto", "max_tokens": 200, "temperature": 0.0}
    pick_ok = False
    try:
        out, wall = call(base, body)
        ch = out["choices"][0]
        tcs = ch["message"].get("tool_calls") or []
        pick = tcs[0]["function"]["name"] if tcs else None
        try:
            margs = json.loads(tcs[0]["function"]["arguments"]) if tcs else None
        except Exception:
            margs = None
        pick_ok = pick == "get_time" and isinstance(margs, dict)
        print(f"MULTITOOL picked={pick} args={json.dumps(margs)} finish={ch.get('finish_reason')} wall={wall:.1f}s")
    except Exception as exc:
        print(f"MULTITOOL error={str(exc)[:120]}")
    print(f"MULTITOOL_VERDICT={'CORRECT_PICK' if pick_ok else 'CHECK_RAW'}")

    # nested arguments
    body = {"model": a.model,
            "messages": [{"role": "user", "content": "Create a calendar event titled 'Rig review' on 2026-09-24 "
                                                    "at 10:00 with attendees Alice and Bob. Use the "
                                                    "create_calendar_event tool."}],
            "tools": [TOOL_NESTED], "tool_choice": "auto", "max_tokens": 300, "temperature": 0.0}
    nested_ok = False
    try:
        out, wall = call(base, body)
        ch = out["choices"][0]
        tcs = ch["message"].get("tool_calls") or []
        if tcs:
            try:
                args = json.loads(tcs[0]["function"]["arguments"])
                ev = args.get("event") or {}
                nested_ok = (tcs[0]["function"]["name"] == "create_calendar_event"
                             and isinstance(ev.get("title"), str) and "rig review" in ev["title"].lower()
                             and isinstance(ev.get("attendees"), list) and len(ev["attendees"]) == 2
                             and isinstance(ev.get("when"), dict)
                             and "2026-09-24" in str(ev["when"].get("date", ""))
                             and "10:00" in str(ev["when"].get("time", "")))
            except Exception:
                nested_ok = False
        print(f"NESTED_VERDICT={'PASS' if nested_ok else 'FAIL'} wall={wall:.1f}s")
    except Exception as exc:
        print(f"NESTED error={str(exc)[:120]}")
        print("NESTED_VERDICT=FAIL")

    ok = not fails and pick_ok and nested_ok
    print(f"TOOLCALL {'PASS' if ok else 'FAIL'}"
          f"{'' if ok else '  [single=' + str(a.count - len(fails)) + '/' + str(a.count) +
             ' pick=' + ('ok' if pick_ok else 'bad') + ' nested=' + ('ok' if nested_ok else 'bad') + ']'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
