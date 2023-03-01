#                                                                       Modules
# =============================================================================

# Third party
from turtle import color

import matplotlib.pyplot as plt
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Jiaxiang Yi (J.Yi@tudelft.nl)"
__credits__ = ["Jiaxiang Yi"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


class PlotRVE2D:
    @staticmethod
    def cricle_inclusion_plot(
        circle_position: np.ndarray,
        radius: float,
        len_start: float,
        len_end: float,
        wid_start: float,
        wid_end: float,
        vol_frac: float,
        save_figure: bool = False,
        fig_name: str = "RVE.png",
    ) -> None:
        """vistualize the 2D RVE

        Parameters
        ----------
        circle_position : np.ndarray
            center position of fibers
        radius : float
            radius of fibers
        len_start : float
            start length of RVE modeling in Abaqus
        len_end : float
            end length of RVE modeling in Abaqus
        wid_start : float
            start width of RVE modeling in Abaqus
        wid_end : float
            end width of RVE modeling in Abaqus
        vol_frac : float
            Actual volume fraction
        save_figure : bool, optional
            save figure , by default False
        """
        # number of inclusions
        num_circles = circle_position.shape[0]
        figure, axes = plt.subplots()
        for ii in range(num_circles):
            if circle_position[ii, 2] == 1:
                cc = plt.Circle(
                    (circle_position[ii, 0], circle_position[ii, 1]),
                    radius,
                    color="lightseagreen",
                )
            elif circle_position[ii, 2] == 2:
                cc = plt.Circle(
                    (circle_position[ii, 0], circle_position[ii, 1]),
                    radius,
                    color="orange",
                )
            elif circle_position[ii, 2] == 4:
                cc = plt.Circle(
                    (circle_position[ii, 0], circle_position[ii, 1]),
                    radius,
                    color="firebrick",
                )
            else:
                raise ValueError("Spltting number is wrong!  \n")
            axes.add_artist(cc)
        axes.set_aspect(1)
        plt.vlines(
            x=len_start + radius,
            ymin=wid_start + radius,
            ymax=wid_end - radius,
            colors="red",
            ls="--",
        )
        plt.vlines(
            x=len_end - radius,
            ymin=wid_start + radius,
            ymax=wid_end - radius,
            colors="red",
            ls="--",
        )
        plt.hlines(
            xmin=len_start + radius,
            xmax=len_end - radius,
            y=wid_end - radius,
            colors="red",
            ls="--",
        )
        plt.hlines(
            xmin=len_start + radius,
            xmax=len_end - radius,
            y=wid_start + radius,
            colors="red",
            ls="--",
        )
        plt.vlines(
            x=len_start, ymin=wid_start, ymax=wid_end, colors="green", ls=":"
        )
        plt.vlines(
            x=len_end, ymin=wid_start, ymax=wid_end, colors="green", ls=":"
        )
        plt.hlines(
            xmin=len_start, xmax=len_end, y=wid_end, colors="green", ls=":"
        )
        plt.hlines(
            xmin=len_start, xmax=len_end, y=wid_start, colors="green", ls=":"
        )
        plt.xlim((len_start - radius, len_end + radius))
        plt.ylim((wid_start - radius, wid_end + radius))
        plt.title(f"$V_f$ = {vol_frac*100:.2f}")
        if not save_figure:
            plt.show()
            plt.close()
        else:
            plt.savefig(fig_name, dpi=300, bbox_inches="tight")
            plt.close()

    @staticmethod
    def heter_cricle_inclusion_plot(
        circle_position: np.ndarray,
        radius_mu: float,
        len_start: float,
        len_end: float,
        wid_start: float,
        wid_end: float,
        vol_frac: float,
        save_figure: bool = False,
        fig_name: str = "RVE.png",
    ) -> None:
        """_summary_

        Parameters
        ----------
        circle_position : np.ndarray
            _description_
        len_start : float
            _description_
        len_end : float
            _description_
        wid_start : float
            _description_
        wid_end : float
            _description_
        vol_frac : float
            _description_
        save_figure : bool, optional
            _description_, by default False
        fig_name : str, optional
            _description_, by default "RVE.png"
        """
        num_circles = circle_position.shape[0]
        figure, axes = plt.subplots()
        for ii in range(num_circles):
            if circle_position[ii, 3] == 1:
                cc = plt.Circle(
                    (circle_position[ii, 0], circle_position[ii, 1]),
                    circle_position[ii, 2],
                    color="lightseagreen",
                )
            elif circle_position[ii, 3] == 2:
                cc = plt.Circle(
                    (circle_position[ii, 0], circle_position[ii, 1]),
                    circle_position[ii, 2],
                    color="orange",
                )
            elif circle_position[ii, 3] == 4:
                cc = plt.Circle(
                    (circle_position[ii, 0], circle_position[ii, 1]),
                    circle_position[ii, 2],
                    color="firebrick",
                )
            else:
                raise ValueError("Spltting number is wrong!  \n")
            axes.add_artist(cc)
        axes.set_aspect(1)
        plt.vlines(
            x=len_start + radius_mu,
            ymin=wid_start + radius_mu,
            ymax=wid_end - radius_mu,
            colors="red",
            ls="--",
        )
        plt.vlines(
            x=len_end - radius_mu,
            ymin=wid_start + radius_mu,
            ymax=wid_end - radius_mu,
            colors="red",
            ls="--",
        )
        plt.hlines(
            xmin=len_start + radius_mu,
            xmax=len_end - radius_mu,
            y=wid_end - radius_mu,
            colors="red",
            ls="--",
        )
        plt.hlines(
            xmin=len_start + radius_mu,
            xmax=len_end - radius_mu,
            y=wid_start + radius_mu,
            colors="red",
            ls="--",
        )
        plt.vlines(
            x=len_start, ymin=wid_start, ymax=wid_end, colors="green", ls=":"
        )
        plt.vlines(
            x=len_end, ymin=wid_start, ymax=wid_end, colors="green", ls=":"
        )
        plt.hlines(
            xmin=len_start, xmax=len_end, y=wid_end, colors="green", ls=":"
        )
        plt.hlines(
            xmin=len_start, xmax=len_end, y=wid_start, colors="green", ls=":"
        )
        plt.xlim((len_start - radius_mu, len_end + radius_mu))
        plt.ylim((wid_start - radius_mu, wid_end + radius_mu))
        plt.title(f"$V_f$ = {vol_frac*100:.2f}")
        if not save_figure:
            plt.show()
            plt.close()
        else:
            plt.savefig(fig_name, dpi=300, bbox_inches="tight")
            plt.close()


class PlotRVE3D:
    @staticmethod
    def sphere_coordinate(location_information: np.ndarray) -> tuple:
        """generate coordinate of sphere

        Parameters
        ----------
        location_information : np.ndarray
           a numpy array contains center and radius info of
           a sphere [x, y, z, r]

        Returns
        -------
        tuple
            coordinate of x, y, z
        """

        loc_info = np.reshape(location_information, (1, -1))
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = loc_info[0, 3] * np.outer(np.cos(u), np.sin(v)) + loc_info[0, 0]
        y = loc_info[0, 3] * np.outer(np.sin(u), np.sin(v)) + loc_info[0, 1]
        z = (
            loc_info[0, 3] * np.outer(np.ones(np.size(u)), np.cos(v))
            + loc_info[0, 2]
        )

        return x, y, z

    def heter_radius_sphere_plot(
        self,
        location_information: np.ndarray,
        radius_mu: float,
        length: float,
        width: float,
        height: float,
        vol_frac: float,
        save_figure: bool = False,
        fig_name: str = "cubic.png",
    ) -> None:
        """plot 3d rve with sphere inclusion

        Parameters
        ----------
        location_information : np.ndarray
            location information
        length : float
            length of cubic
        width : float
            width of cubic
        height : float
            height of cubic
        vol_frac : float
            volume fraction
        save_figure : bool, optional
            a flag to indicate if we want save the figure or not, by default False
        fig_name : str, optional
            fig name, by default "cubic_rve.png"
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # Plot the surface
        for ii in range(location_information.shape[0]):
            x, y, z = self.sphere_coordinate(
                location_information=location_information[ii, :]
            )
            if location_information[ii, 4] == 1:
                ax.plot_surface(x, y, z, color="lightseagreen")
            elif location_information[ii, 4] == 2:
                ax.plot_surface(x, y, z, color="orange")
            elif location_information[ii, 4] == 4:
                ax.plot_surface(x, y, z, color="blue")
            elif location_information[ii, 4] == 8:
                ax.plot_surface(x, y, z, color="firebrick")
            else:
                raise ValueError("Spltting number is wrong!\n")
        axes = [int(length), int(width), int(height)]
        data = np.ones(axes, dtype=np.bool)
        alpha = 0.6
        colors = np.empty(axes + [4], dtype=np.float32)
        colors[:] = [1, 1, 1, alpha]
        ax.voxels(data, facecolors=colors)
        ax.set_xlim3d(-2 * radius_mu, length + 2 * radius_mu)
        ax.set_ylim3d(-2 * radius_mu, width + 2 * radius_mu)
        ax.set_zlim3d(-2 * radius_mu, height + 2 * radius_mu)
        plt.title(f"$V_f$ = {vol_frac*100:.2f}")

        # Set an equal aspect ratio
        ax.set_aspect("auto")
        if not save_figure:
            plt.show()

        else:
            plt.savefig(fig_name, dpi=300, bbox_inches="tight")
            plt.close()
