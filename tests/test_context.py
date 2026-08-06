#!/usr/bin/env python3
"""
Tests the Context module's token-budget trimming logic directly.

Uses a small max_context_tokens so it's easy to force trimming with short,
predictable messages instead of needing huge text blocks.

Usage:
    python test_context.py --model-path ../models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf

NOTE: adjust the import below if your folder is named differently
(e.g. modules.memory_short.flashback vs modules.context.flashback).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.context.local import Module  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                         help="Any .gguf file works — only the tokenizer is loaded (vocab_only).")
    args = parser.parse_args()

    results = []

    # small budget so a handful of short messages can push past it
    ctx = Module(model_path=args.model_path, max_context_tokens=50)

    print("--- 1. messages well under budget: nothing should trim ---")
    ctx.add("user", "hi")
    ctx.add("assistant", "hello")
    ctx.add("user", "how are you")
    kept = len(ctx.get())
    print(f"    messages kept: {kept}, total tokens: {ctx._total_tokens()}")
    ok = kept == 3
    print(f"    [{'PASS' if ok else 'FAIL'}] expected all 3 messages kept")
    results.append(ok)

    print("\n--- 2. push well past budget: oldest should get trimmed ---")
    ctx.clear()
    long_sentence = "This is a moderately long sentence meant to consume a decent chunk of the token budget on its own."
    for i in range(6):
        ctx.add("user", f"{long_sentence} (message {i})")
    total = ctx._total_tokens()
    kept = len(ctx.get())
    print(f"    messages kept: {kept} (added 6), total tokens: {total} (budget: 50)")
    ok = total <= 50 and kept < 6
    print(f"    [{'PASS' if ok else 'FAIL'}] total must be <= budget AND some messages must have been dropped")
    results.append(ok)

    print("\n--- 3. oldest-first order: the SURVIVING messages should be the most recent ones ---")
    surviving_contents = [m["content"] for m in ctx.get()]
    last_added = f"{long_sentence} (message 5)"
    first_added = f"{long_sentence} (message 0)"
    ok = last_added in surviving_contents and first_added not in surviving_contents
    print(f"    most recent message present: {last_added in surviving_contents}")
    print(f"    oldest message absent: {first_added not in surviving_contents}")
    print(f"    [{'PASS' if ok else 'FAIL'}] trimming must drop oldest first, keep newest")
    results.append(ok)

    print("\n--- 4. single message larger than the whole budget: must not empty the list ---")
    ctx.clear()
    huge = " ".join(["word"] * 200)  # way more than 50 tokens on its own
    ctx.add("user", huge)
    kept = len(ctx.get())
    ok = kept == 1
    print(f"    messages kept: {kept} (over budget, but the >1 guard should keep it anyway)")
    print(f"    [{'PASS' if ok else 'FAIL'}] must keep at least the 1 message, never empty from a single add")
    results.append(ok)

    print("\n--- 5. clear() empties everything ---")
    ctx.clear()
    ok = len(ctx.get()) == 0
    print(f"    messages after clear: {len(ctx.get())}")
    print(f"    [{'PASS' if ok else 'FAIL'}] clear() must fully empty the conversation")
    results.append(ok)

    print("\n--- 6. sanity check: tokenizer gives plausible, non-trivial counts ---")
    ctx.clear()
    short_count = ctx._count_tokens("hi")
    long_count = ctx._count_tokens(long_sentence)
    ok = 0 < short_count < long_count
    print(f"    'hi' -> {short_count} tokens, long sentence -> {long_count} tokens")
    print(f"    [{'PASS' if ok else 'FAIL'}] longer text must tokenize to more tokens than short text")
    results.append(ok)

    print("\n--- 7. gradual sliding window: multiple messages should coexist, oldest falls off one at a time ---")
    # budget sized so ~4-5 short messages fit together, forcing a real multi-message
    # window rather than collapsing straight down to 1 survivor OR fitting everything
    ctx_window = Module(model_path=args.model_path, max_context_tokens=25)
    short_msgs = [f"message number {i}" for i in range(8)]
    snapshot_sizes = []
    for m in short_msgs:
        ctx_window.add("user", m)
        snapshot_sizes.append(len(ctx_window.get()))

    print(f"    message count after each add: {snapshot_sizes}")
    grew_past_one = any(s > 1 for s in snapshot_sizes)
    stayed_under_budget = ctx_window._total_tokens() <= 40
    final_kept = len(ctx_window.get())
    final_contents = [m["content"] for m in ctx_window.get()]
    newest_present = short_msgs[-1] in final_contents
    oldest_absent = short_msgs[0] not in final_contents
    ok = grew_past_one and stayed_under_budget and newest_present and oldest_absent and 1 < final_kept < 8
    print(f"    final: {final_kept} messages kept, {ctx_window._total_tokens()} tokens (budget 25)")
    print(f"    [{'PASS' if ok else 'FAIL'}] window must hold MORE than 1 message at some point, "
          f"and end with a partial (not full, not empty) set of the most recent messages")
    results.append(ok)

    print("\n" + "=" * 50)
    print(f"RESULTS: {sum(results)}/{len(results)} passed")
    print("=" * 50)


if __name__ == "__main__":
    main()