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
                print(f"\nUser: {user_input}")
                output.interrupt()

                memories = memory.retrieve(user_input)
                context = contextMgr.get()
                reply = ""

                try:
                    for token in core.generate(user_input, context, memories):
                        if inp.has_input():
                            core.interrupt()
                            output.interrupt()
                            break
                        output.send(token)
                        reply += token
                except KeyboardInterrupt:
                    core.interrupt()
                    output.interrupt()
                    print("\n[interrupted]")
            
                contextMgr.add("user", user_input)
                contextMgr.add("assistant", reply)
                memory.store([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply}
                ])

        except KeyboardInterrupt:
            print("\nExiting (Ctrl+C).")
