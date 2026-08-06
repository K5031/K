import signal
from interfaces import ControllerInterface
from config import system_prompt


class Module(ControllerInterface):
    required_modules = ["input", "output", "core", "context", "memory"]

    def run(self, api) -> None:
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

                memories = memory.retrieve(user_input)
                context = contextMgr.get()
                reply = ""

                interrupted = {"flag": False}

                def _on_sigint(signum, frame):
                    interrupted["flag"] = True

                previous_handler = signal.signal(signal.SIGINT, _on_sigint)
                try:
                    for token in core.generate(user_input, context, memories):
                        if inp.has_input() or interrupted["flag"]:
                            core.interrupt()
                            output.interrupt()
                            interrupted["flag"] = False
                            print("\n[interrupted]")
                            break

                        output.send(token)
                        reply += token
                    else:
                        if hasattr(output, "flush"):
                            output.flush()
                finally:
                    signal.signal(signal.SIGINT, previous_handler)

                contextMgr.add("user", user_input)
                contextMgr.add("assistant", reply)
                memory.store([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply}
                ])

        except KeyboardInterrupt:
            print("\nExiting (Ctrl+C).")