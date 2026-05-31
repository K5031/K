from interfaces import InputInterface

class Module(InputInterface):
    def get_input(self):
        return input("\n> ")

    def has_input(self) -> bool:
        return False