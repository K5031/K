#!/usr/bin/env python3
"""
Benchmark: KV cache on GPU (offload_kqv=True, default) vs KV cache in RAM
(offload_kqv=False) for llama_cpp.server.

Usage:
    python bench_kv_offload.py --model /path/to/model.gguf --n-gpu-layers 99

Spins up the server twice (once per KV cache mode), sends the same prompt
a few times to each, and reports tokens/sec + time-to-first-token so you
can see the real speed cost of RAM-offloaded KV cache on your hardware.
"""

import argparse
import json
import os
import signal
import subprocess
import time

import requests

PORT = 8099  # separate port so it doesn't clash with your app's running server
BASE_URL = f"http://localhost:{PORT}"

PROMPT = (
    "Write a short paragraph explaining how photosynthesis works, "
    "covering the role of chlorophyll, sunlight, water, and carbon dioxide."
)


def start_server(model_path, n_gpu_layers, n_ctx, offload_kqv, flash_attn):
    args = [
        "python", "-m", "llama_cpp.server",
        "--model", model_path,
        "--port", str(PORT),
        "--n_gpu_layers", str(n_gpu_layers),
        "--n_ctx", str(n_ctx),
        "--offload_kqv", str(offload_kqv),
        "--flash_attn", str(flash_attn),
        "--verbose", "False",
    ]
    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_server(timeout=90):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/v1/models")
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError("server failed to start within timeout")


def stop_server(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_completion(prompt, max_tokens):
    """Streams a completion, timing first-token latency and total throughput."""
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }
    start = time.time()
    first_token_time = None
    token_count = 0

    with requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1

    end = time.time()
    total_time = end - start
    ttft = (first_token_time - start) if first_token_time else None
    tok_per_sec = token_count / total_time if total_time > 0 else 0
    return {
        "tokens": token_count,
        "total_time": total_time,
        "ttft": ttft,
        "tokens_per_sec": tok_per_sec,
    }


def benchmark_mode(model_path, n_gpu_layers, n_ctx, offload_kqv, flash_attn, rounds, max_tokens):
    label = "RAM (offload_kqv=False)" if not offload_kqv else "GPU (offload_kqv=True)"
    print(f"\n=== Starting server: KV cache on {label} ===")
    proc = start_server(model_path, n_gpu_layers, n_ctx, offload_kqv, flash_attn)
    try:
        wait_for_server()
        results = []
        for i in range(rounds):
            print(f"  round {i + 1}/{rounds}...")
            r = run_completion(PROMPT, max_tokens)
            results.append(r)
            print(f"    tokens={r['tokens']}  time={r['total_time']:.2f}s  "
                  f"ttft={r['ttft']:.2f}s  {r['tokens_per_sec']:.1f} tok/s")
        return results
    finally:
        stop_server(proc)
        time.sleep(2)  # let the port free up before the next server starts


def summarize(label, results):
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_ttft = sum(r["ttft"] for r in results if r["ttft"]) / len(results)
    print(f"\n{label}: avg {avg_tps:.1f} tok/s, avg time-to-first-token {avg_ttft:.2f}s")
    return avg_tps, avg_ttft


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache on GPU vs RAM")
    parser.add_argument("--model", required=True, help="Path to .gguf model")
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--flash-attn", default="True")
    parser.add_argument("--rounds", type=int, default=3, help="Completions per mode")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    gpu_results = benchmark_mode(
        args.model, args.n_gpu_layers, args.n_ctx,
        offload_kqv=True, flash_attn=args.flash_attn,
        rounds=args.rounds, max_tokens=args.max_tokens,
    )
    ram_results = benchmark_mode(
        args.model, args.n_gpu_layers, args.n_ctx,
        offload_kqv=False, flash_attn=args.flash_attn,
        rounds=args.rounds, max_tokens=args.max_tokens,
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    gpu_tps, gpu_ttft = summarize("KV cache on GPU ", gpu_results)
    ram_tps, ram_ttft = summarize("KV cache in RAM ", ram_results)

    slowdown = (gpu_tps / ram_tps) if ram_tps else float("inf")
    print(f"\nGPU-cache is {slowdown:.2f}x faster than RAM-offloaded cache on this hardware.")


if __name__ == "__main__":
    main()