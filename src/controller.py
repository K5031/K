import signal
import time
from registry import Registry
from config import system_prompt


def run():
    with Registry.from_config() as api:
        inp = api.get("input")
        output = api.get("output")
        core = api.get("core")
        contextMgr = api.get("context")
        memory = api.get("memory")

        core.set_system_prompt(system_prompt)

        print(f"Selected model: {core.model_path}")

        try:
            while True:
                user_input = inp.get_input()
                output.interrupt()
                turn_start = time.time()

                t0 = time.time()
                memories = memory.retrieve(user_input)
                t1 = time.time()
                print(f"[timing] retrieve()        {t1 - t0:.2f}s")

                context = contextMgr.get()
                t2 = time.time()
                print(f"[timing] context.get()     {t2 - t1:.2f}s")

                reply = ""
                first_token_time = None
                token_count = 0

                # SIGINT handling only matters DURING generation — a Ctrl+C
                # can land inside llama.cpp's C-level log callback, where a
                # raised KeyboardInterrupt gets silently swallowed by ctypes
                # instead of propagating back to Python. Using a flag set by
                # a custom handler (rather than relying on the exception)
                # sidesteps that. Scoped to just this block so Ctrl+C during
                # inp.get_input() keeps its normal, reliable behavior.
                interrupted = {"flag": False}

                def _on_sigint(signum, frame):
                    interrupted["flag"] = True

                previous_handler = signal.signal(signal.SIGINT, _on_sigint)
                try:
                    gen_start = time.time()
                    for token in core.generate(user_input, context, memories):
                        if first_token_time is None:
                            first_token_time = time.time()
                            print(f"[timing] time-to-first-token {first_token_time - gen_start:.2f}s")

                        if inp.has_input() or interrupted["flag"]:
                            core.interrupt()
                            output.interrupt()
                            print("\n[interrupted]")
                            break

                        output.send(token)
                        reply += token
                        token_count += 1

                    gen_end = time.time()
                    gen_total = gen_end - gen_start
                    tok_per_sec = token_count / gen_total if gen_total > 0 else 0
                    print(f"[timing] generate() total  {gen_total:.2f}s "
                          f"({token_count} tokens, {tok_per_sec:.1f} tok/s)")
                finally:
                    signal.signal(signal.SIGINT, previous_handler)

                t3 = time.time()
                contextMgr.add("user", user_input)
                contextMgr.add("assistant", reply)
                t4 = time.time()
                print(f"[timing] context.add() x2  {t4 - t3:.2f}s")

                memory.store([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply}
                ])
                t5 = time.time()
                print(f"[timing] store() (queued)  {t5 - t4:.2f}s")

                turn_end = time.time()
                print(f"[timing] === total turn ===  {turn_end - turn_start:.2f}s\n")

        except KeyboardInterrupt:
            print("\nExiting (Ctrl+C).")