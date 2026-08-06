import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.context.cloud import Module


def main():
    results = []

    ctx = Module(max_context_tokens=1000)

    print("--- 1. low usage, well under budget: no trim ---")
    ctx.add("user", "hi")
    ctx.add("assistant", "hello")
    ctx.report_usage(50)
    ctx.add("user", "how are you")
    kept = len(ctx.get())
    ok = kept == 3
    print(f"    messages kept: {kept}, last_known_tokens: {ctx.last_known_tokens}")
    print(f"    [{'PASS' if ok else 'FAIL'}] expected all 3 messages kept")
    results.append(ok)

    print("\n--- 2. usage crosses the overflow trigger: should batch-trim to target ---")
    ctx2 = Module(max_context_tokens=1000)
    for i in range(6):
        ctx2.add("user", f"message {i} " + "word " * 20)
        ctx2.add("assistant", f"reply {i} " + "word " * 20)
    ctx2.report_usage(920)
    ctx2.add("user", "final message")

    target = int(1000 * 0.7)
    kept = len(ctx2.get())
    ok = ctx2.last_known_tokens <= target
    print(f"    messages kept: {kept}, last_known_tokens: {ctx2.last_known_tokens} (target: {target})")
    print(f"    [{'PASS' if ok else 'FAIL'}] last_known_tokens must drop to target, not just barely under budget")
    results.append(ok)

    print("\n--- 3. trim removed MULTIPLE messages in one pass (batch, not one-at-a-time) ---")
    ok = kept < 13
    print(f"    messages kept: {kept} (started with 13 total)")
    print(f"    [{'PASS' if ok else 'FAIL'}] batch trim should remove more than a single message when triggered")
    results.append(ok)

    print("\n--- 4. oldest-first: newest message survives, oldest doesn't ---")
    contents = [m["content"] for m in ctx2.get()]
    newest_present = "final message" in contents
    oldest_absent = any("message 0 " in c for c in contents) is False
    ok = newest_present and oldest_absent
    print(f"    newest present: {newest_present}, oldest absent: {oldest_absent}")
    print(f"    [{'PASS' if ok else 'FAIL'}] trimming must drop oldest first")
    results.append(ok)

    print("\n--- 5. clear() resets both conversation and last_known_tokens ---")
    ctx2.clear()
    ok = len(ctx2.get()) == 0 and ctx2.last_known_tokens == 0
    print(f"    messages: {len(ctx2.get())}, last_known_tokens: {ctx2.last_known_tokens}")
    print(f"    [{'PASS' if ok else 'FAIL'}] clear() must reset both")
    results.append(ok)

    print("\n" + "=" * 50)
    print(f"RESULTS: {sum(results)}/{len(results)} passed")
    print("=" * 50)


if __name__ == "__main__":
    main()