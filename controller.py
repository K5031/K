from registry import Registry

def run():
    with open("system_prompt.txt") as f:
        system_prompt = f.read()

    with Registry.from_config() as api:
        inp = api.get("input")
        output = api.get("output")
        core = api.get("core")
        short_mem = api.get("memory_short")
        long_mem = api.get("memory_long")

        core.set_system_prompt(system_prompt)

        print(f"Selected model: {core.model_path}")

        try:
            while True:
                user_input = inp.get_input()
                print(f"\nUser: {user_input}")
                output.interrupt()

                memories = long_mem.retrieve(user_input)
                context = short_mem.get()
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
            
                short_mem.add("user", user_input)
                short_mem.add("assistant", reply)
                long_mem.store([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply}
                ])

        except KeyboardInterrupt:
            print("\nExiting (Ctrl+C).")
