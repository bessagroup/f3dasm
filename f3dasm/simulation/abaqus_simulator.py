from ..base.simulation import Simulator


class AbaqusSimulator(Simulator):
    def pre_process(self) -> None:
        """Function that handles the Abaqus pre-processing"""
        pass

    def execute(self) -> None:
        """Function that calls the FEM simulator the Abaqus pre-processing"""
        pass

    def post_process(self) -> None:
        """Function that handles the Abaqus post-processing"""
        pass
