from interfaces import OutputInterface

class Module(OutputInterface):
    def send(self, token: str) -> None:
        print(token, end="", flush=True)
    
    def interrupt(self):
        pass