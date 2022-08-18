class Simulator:
    """Base class for a FEM simulator"""

    def __init__():
        pass

    def pre_process(self):
        pass

    def execute(self):
        pass

    def post_process(self):
        pass

    def run(self) -> None:
        self.pre_process()
        self.execute()
        self.post_process()
