#                                                                       Modules
# =============================================================================

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Jiaxiang Yi (J.Yi@tudelft.nl)"
__credits__ = ["Jiaxiang Yi"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


class MicrosctucturaGenerator:
    def generate_rve(self) -> float:
        """function that used to generate the geometry information for
        Abaqus"""

        raise NotImplementedError(
            "The function should be implemented in sub-class \n"
        )

    def plot_rve(
        self, save_figure: bool = False, fig_name: str = "RVE.png"
    ) -> None:
        """plot figure for RVE

        Parameters
        ----------
        save_figure : bool, optional
            save figure or not , by default False
        fig_name : str, optional
            figure name, by default "RVE.png"

        Raises
        ------
        NotImplementedError
            error report
        """

        raise NotImplementedError(
            "The function should be implemented in sub-class \n"
        )

    def save_results(
        self, file_name: str = "micro_structure_info.json"
    ) -> None:
        """save results

        Parameters
        ----------
        file_name : str, optional
            file name, by default "micro_structure_info.json"

        Raises
        ------
        NotImplementedError
            error report
        """

        raise NotImplementedError(
            "The function should be implemented in sub-class \n"
        )
