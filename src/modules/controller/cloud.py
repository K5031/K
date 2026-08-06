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
        print(f"Selected model: {core.model}")

        try:
            while True:
                user_input = inp.get_input()
                output.interrupt()

                memories = memory.retrieve(user_input)
                context = contextMgr.get()
                reply = ""

                try:
                    for token in core.generate(user_input, context, memories):
                        if inp.has_input():
                            core.interrupt()
                            output.interrupt()
                            print("\n[interrupted]")
                            break

                        output.send(token)
                        reply += token
                    else:
                        if hasattr(output, "flush"):
                            output.flush()
                except KeyboardInterrupt:
                    core.interrupt()
                    output.interrupt()
                    print("\n[interrupted]")

                if core.last_usage:
                    contextMgr.report_usage(core.last_usage["prompt_tokens"])

                contextMgr.add("user", user_input)
                contextMgr.add("assistant", reply)
                memory.store([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply}
                ])

        except KeyboardInterrupt:
            print("\nExiting (Ctrl+C).")