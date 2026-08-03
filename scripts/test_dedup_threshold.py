#!/usr/bin/env python3
"""
Quick calibration check for Remis's dedup_distance threshold.

Embeds pairs of sentences — some that SHOULD be treated as duplicates
(paraphrases of the same fact) and some that SHOULD NOT (genuinely
different facts) — and prints the cosine distance between each pair.

This tells you whether the current dedup_distance (0.15) is set correctly:
- "should dedupe" pairs should score BELOW the threshold
- "should NOT dedupe" pairs should score ABOVE the threshold

Usage: python test_dedup_threshold.py
"""

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CURRENT_THRESHOLD = 0.2

PAIRS = [
    # (label, sentence_a, sentence_b, should_dedupe)
    ("exact restatement", "User's name is K5031", "User's name is K5031", True),
    ("paraphrase (name)", "User's name is K5031", "User is called K5031", True),
    ("paraphrase (name 2)", "User's name is K5031", "The user's name is K5031", True),
    ("different phrasing (job)", "User works as a teacher", "User's job is teaching", True),
    ("genuinely different", "User's name is K5031", "User goes to UCL", False),
    ("genuinely different 2", "User works as a teacher", "User has a dog named Rex", False),
    ("related but distinct", "User goes to UCL", "User studies computer science at UCL", False),
    ("contradicting fact", "User's name is K5031", "User's name is Alex", True),  # tricky: should this dedupe or not?
]


def main():
    print(f"Loading embedder: {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    print(f"Current dedup_distance threshold: {CURRENT_THRESHOLD}\n")

    print(f"{'label':<28} {'distance':>9}  {'expected':<10} {'result'}")
    print("-" * 70)

    correct = 0
    for label, a, b, should_dedupe in PAIRS:
        emb_a = model.encode(a)
        emb_b = model.encode(b)
        distance = cosine(emb_a, emb_b)  # scipy cosine() returns 1 - cosine_similarity

        would_dedupe = distance <= CURRENT_THRESHOLD
        expected_str = "dedupe" if should_dedupe else "keep both"
        actual_str = "would dedupe" if would_dedupe else "would keep both"
        is_correct = would_dedupe == should_dedupe
        correct += is_correct
        mark = "OK" if is_correct else "MISMATCH"

        print(f"{label:<28} {distance:>9.4f}  {expected_str:<10} -> {actual_str}  [{mark}]")
        if label == "contradicting fact":
            print("   ^ NOTE: near-identical embedding but CONTRADICTS the old fact —")
            print("     dedup treats this as 'same', so it silently keeps the OLD name")
            print("     and drops the correction. Worth deciding if that's acceptable.")

    print(f"\n{correct}/{len(PAIRS)} matched expected behavior at threshold={CURRENT_THRESHOLD}")
    print("\nIf mismatches are common, try adjusting CURRENT_THRESHOLD above and rerun")
    print("to find a value that separates 'paraphrase' pairs from 'genuinely different' pairs.")


if __name__ == "__main__":
    main()