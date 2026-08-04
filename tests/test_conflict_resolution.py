#!/usr/bin/env python3
"""
End-to-end test of Remis's conflict resolution — drives the REAL Module
class (spawns its own server, does real extraction, real dedup, real
conflict-check calls) against a throwaway test collection so it never
touches your actual memories.

Usage:
    python test_conflict_resolution.py --model-path Qwen3.5-9B-Q4_K_M.gguf --n-gpu-layers 32

Run from src/scripts/ (same place as your other test/utility scripts) —
it imports Module the same way your app does.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.memory.remis import Module


def get_all_facts(remis) -> list[str]:
    results = remis.collection.get(include=["metadatas"])
    return [m.get("data", "") for m in results["metadatas"]]


def run_test(name, remis, user_msg, expected_facts_after, description):
    print(f"\n--- {name} ---")
    print(f"    input: \"{user_msg}\"")
    remis._extract_and_save([
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "Got it."},
    ])
    facts = get_all_facts(remis)
    print(f"    DB now contains ({len(facts)}): {facts}")

    status = "PASS" if len(facts) == expected_facts_after else "FAIL"
    print(f"    [{status}] expected {expected_facts_after} total fact(s) — {description}")
    return status == "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-gpu-layers", type=int, default=32)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8091)  # separate from your real Remis port
    args = parser.parse_args()

    print("Starting a throwaway Remis instance (test collection, won't touch real memories)...")
    remis = Module(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        collection_name="remis_conflict_test",
        port=args.port,
    )

    # start clean, in case a previous run left data behind
    try:
        existing = remis.collection.get()
        if existing["ids"]:
            remis.collection.delete(ids=existing["ids"])
    except Exception:
        pass

    results = []

    results.append(run_test(
        "1. store initial fact",
        remis, "my name is K5031",
        expected_facts_after=1,
        description="first fact stored cleanly",
    ))

    results.append(run_test(
        "2. exact restatement (should dedupe)",
        remis, "my name is K5031",
        expected_facts_after=1,
        description="restating the same fact should NOT add a duplicate",
    ))

    results.append(run_test(
        "3. genuine correction (should replace, not duplicate)",
        remis, "actually my name is Alex",
        expected_facts_after=1,
        description="correcting the name should REPLACE the old entry, not add a 2nd",
    ))

    facts_after_correction = get_all_facts(remis)
    correct_value = any("Alex" in f for f in facts_after_correction) and \
        not any("K5031" in f for f in facts_after_correction)
    print(f"    [{'PASS' if correct_value else 'FAIL'}] the surviving fact actually says Alex, not K5031")
    results.append(correct_value)

    results.append(run_test(
        "4. related but distinct (should NOT be treated as conflict)",
        remis, "I go to UCL",
        expected_facts_after=2,
        description="new unrelated fact should just be added, total now 2",
    ))

    results.append(run_test(
        "5. related-but-distinct pair (should stay separate, not merge)",
        remis, "I study computer science at UCL",
        expected_facts_after=3,
        description="close to fact 4 but adds new info — should be kept as its own 3rd entry, not merged",
    ))

    results.append(run_test(
        "6. false-positive risk: different SUBJECT, same value",
        remis, "my friend's name is Alex",
        expected_facts_after=4,
        description="a friend named Alex must NOT overwrite the user's own name (also Alex) — "
                    "different subject, same surface value",
    ))

    facts_after_case6 = get_all_facts(remis)
    user_still_alex = any("Alex" in f and "friend" not in f.lower() for f in facts_after_case6)
    friend_alex_added = any("Alex" in f and "friend" in f.lower() for f in facts_after_case6)
    case6_content_ok = user_still_alex and friend_alex_added
    print(f"    [{'PASS' if case6_content_ok else 'FAIL'}] both 'user is Alex' AND "
          f"'friend is Alex' should coexist as separate facts")
    print(f"    (note: this only truly exercises the conflict-check LLM call if the "
          f"embedding distance landed in the conflict band — check the [Remis] logs above "
          f"to confirm a conflict-check actually ran, rather than the pair simply scoring "
          f"as clearly distinct on distance alone)")
    results.append(case6_content_ok)

    results.append(run_test(
        "7. multi-fact turn: one conflicting + one genuinely new",
        remis, "actually my name is Jordan and I also just adopted a cat named Whiskers",
        expected_facts_after=5,
        description="name Alex->Jordan should REPLACE (net 0), cat fact should ADD (+1) — "
                    "tests splitting and conflict-checking working together in one turn",
    ))

    facts_after_case7 = get_all_facts(remis)
    jordan_present = any("Jordan" in f for f in facts_after_case7)
    user_alex_gone = not any("Alex" in f and "friend" not in f.lower() for f in facts_after_case7)
    friend_alex_survived = any("Alex" in f and "friend" in f.lower() for f in facts_after_case7)
    cat_added = any("Whiskers" in f or ("cat" in f.lower() and "Whiskers" not in f) for f in facts_after_case7)
    case7_content_ok = jordan_present and user_alex_gone and friend_alex_survived and cat_added
    print(f"    [{'PASS' if case7_content_ok else 'FAIL'}] Jordan present, user's old Alex gone, "
          f"friend's Alex untouched, cat fact added")
    results.append(case7_content_ok)

    print("\n--- 8. conflict-check call fails (simulated) — must degrade safely ---")
    original_base_url = remis.base_url
    remis.base_url = "http://localhost:1"  # unreachable — forces the request to fail
    conflict_result = remis._check_conflict("User's name is Jordan", "User's name is Sam")
    remis.base_url = original_base_url
    case8_ok = conflict_result == "DISTINCT"
    print(f"    _check_conflict returned: {conflict_result}")
    print(f"    [{'PASS' if case8_ok else 'FAIL'}] on failure it must default to DISTINCT "
          f"(keep both, never silently delete on an error)")
    results.append(case8_ok)

    results.append(run_test(
        "9. paraphrase beyond the tight fast-path (should classify SAME, not duplicate storage)",
        remis, "I'm Jordan",
        expected_facts_after=5,
        description="restating the name differently should land in the conflict-check band "
                    "(too far for the tiny 0.05 fast-path) and get classified SAME by the LLM — "
                    "count must NOT increase",
    ))

    print("\n" + "=" * 50)
    passed = sum(results)
    print(f"RESULTS: {passed}/{len(results)} passed")
    print("=" * 50)

    remis.stop()


if __name__ == "__main__":
    main()