#                                                                       Modules
# =============================================================================
# standard
import json
import math
import time

import matplotlib.pyplot as plt

# Third party
import numpy as np
from scipy.spatial import distance_matrix

# import local functions
from .microstrcuture import MicrosctucturaGenerator
from .microstructure_plots import PlotRVE2D

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Jiaxiang Yi (J.Yi@tudelft.nl)"
__credits__ = ["Jiaxiang Yi"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


class CircleParticles(MicrosctucturaGenerator, PlotRVE2D):
    """This class is used to generate the 2DRVe with different size of
        disks

    Parameters
    ----------
    MicrosctucturaGenerator : class
        parent class of microstructure generater
    PlotRVE2D : class
        vislization of 2D RVE
    """

    def __init__(
        self,
        length: float,
        width: float,
        radius_mu: float,
        radius_std: float,
        vol_req: float,
        num_guess_max: int = 50000,
        num_fiber_max: int = 750,
        num_cycle_max: int = 15,
        dist_min_factor: float = 1.1,
        seed: int = None,
    ) -> None:
        """Initialization

        Parameters
        ----------
        length : float
            length of RVE
        width : float
            width of RVE
        radius_mu : float
            mean of circle's radius
        radius_std : float
            std of circle's radius
        vol_req : float
            required volume fraction
        num_guess_max : int, optional
            maximum guess for fibers, by default 50000
        num_fiber_max : int, optional
            maximum fiber inside RVE, by default 750
        num_cycle_max : int, optional
            iteration cycles, by default 15
        dist_min_factor : float, optional
            distance factor, by default 2.07
        """
        # geometry information of the 2D RVE with homogeneous circles
        self.length = length
        self.width = width
        self.radius_mu = radius_mu
        self.radius_std = radius_std
        self.vol_req = vol_req

        # Initialization of the algorithm
        # Number of guesses in the random generation of fibre position
        self.dist_min_factor = dist_min_factor
        self.num_guess_max = num_guess_max
        self.num_fibers_max = num_fiber_max
        self.num_cycles_max = num_cycle_max

        # decide to use seed or not
        np.random.seed(seed=seed)

    def __parameter_initialization(self) -> None:
        """Initialize the parameters"""
        self.num_change = 3
        self.num_cycle = 0
        self.vol_frac = 0
        # Total volume of image
        self.vol_total = self.length * self.width

        # initial coordinate for position of fibre
        self.len_start = -1 * self.radius_mu
        self.len_end = self.length + self.radius_mu
        self.wid_start = -1 * self.radius_mu
        self.wid_end = self.width + self.radius_mu

        # some import variables
        # fiber location is a nx4 numpy array
        # x, y, r, p (partition)
        self.fiber_positions = None

    def __generate_rve(self) -> None:
        """core procedure of generating RVE"""
        self.__parameter_initialization()
        self._procedure_initialization()
        self._core_iteration()

    def generate_rve(self) -> float:
        """core function for generating RVE microstructures via Melro's
        algorithm

        Returns
        -------
        float
            Actual volume fracture
        """
        start_time = time.time()
        self.__generate_rve()
        end_time = time.time()
        print(
            f"Time of generate the 2D RVE with volume fraction\
            = {self.vol_frac:.2f}  is {(end_time - start_time):.2f} s"
        )

        return self.vol_frac

    def save_results(
        self, file_name: str = "micro_structure_info.json"
    ) -> None:
        """save the needed information of the 2D \
            microstructure into "Json" file.
        Later this Json file will be imported to ABAQUS
        """
        geometry_info = {
            "location_information": self.fiber_positions.tolist(),
            "radius_mu": self.radius_mu,
            "radius_std": self.radius_std,
            "len_start": self.len_start,
            "wid_start": self.wid_start,
            "len_end": self.len_end,
            "wid_end": self.wid_end,
        }
        with open(file_name, "w") as fp:
            json.dump(geometry_info, fp)

    def plot_rve(
        self, save_figure: bool = False, fig_name: str = "RVE.png"
    ) -> None:
        """
        This function is used to visualize the 2D RVE
        """
        self.heter_cricle_inclusion_plot(
            circle_position=self.fiber_positions,
            radius_mu=self.radius_mu,
            len_start=self.len_start,
            len_end=self.len_end,
            wid_start=self.wid_start,
            wid_end=self.wid_end,
            vol_frac=self.vol_frac,
            save_figure=save_figure,
            fig_name=fig_name,
        )

    def _procedure_initialization(self) -> None:
        """This function is used to generate the first disk and
        assign the initial values of the algorithm
        """

        # initialization(generate the first fiber randomly)
        self.fiber_min_dis_vector = np.zeros(
            (self.num_fibers_max, self.num_cycles_max + 1, 2)
        )
        self.num_fibers = 1
        # generate the location of the first fiber
        # the first fiber is generated with one partition
        fiber_temp = self.generate_random_fibers(
            len_start=self.radius_mu,
            len_end=self.length - self.radius_mu,
            wid_start=self.radius_mu,
            wid_end=self.width - self.radius_mu,
            radius_mu=self.radius_mu,
            radius_std=0,
        )
        # update the volume fraction information
        self.vol_frac = self.fiber_volume(self.radius_mu) / self.vol_total
        self.fiber_positions = np.zeros((1, 4))
        self.fiber_positions[0, 0:3] = fiber_temp.T
        self.fiber_positions[0, 3] = 1

    def _core_iteration(self) -> None:
        """core iteration part of the micro-structure generation method"""

        # the main loop of the algorithm
        while (
            self.vol_frac < self.vol_req
            and self.num_cycle < self.num_cycles_max
        ):
            # ================================================================#
            #                   generate the fibers randomly                  #
            # ================================================================#
            self.num_trial = 1
            while (
                self.num_trial < self.num_guess_max
                and self.vol_frac < self.vol_req
                and self.num_fibers < self.num_fibers_max
            ):
                # update the info of number trial
                self.num_trial = self.num_trial + 1
                fiber_temp = self.generate_random_fibers(
                    len_start=0,
                    len_end=self.length,
                    wid_start=0,
                    wid_end=self.width,
                    radius_mu=self.radius_mu,
                    radius_std=self.radius_std,
                )
                # check the location of the fiber and
                new_fiber = self.new_positions(
                    x_center=fiber_temp[0, 0],
                    y_center=fiber_temp[1, 0],
                    radius=fiber_temp[2, 0],
                    length=self.length,
                    width=self.width,
                )
                # check the overlap of new fiber
                overlap_status = self.overlap_check(
                    new_fiber=new_fiber,
                    fiber_pos=self.fiber_positions.copy(),
                    dist_factor=self.dist_min_factor,
                )
                if overlap_status == 0:
                    self.fiber_positions = np.vstack(
                        (self.fiber_positions, new_fiber)
                    )
                    self.vol_frac = (
                        self.vol_frac
                        + self.fiber_volume(new_fiber[0, 2]) / self.vol_total
                    )
                    self.num_fibers = self.num_fibers + new_fiber.shape[0]
                del new_fiber

            # ================================================================#
            #                   striring the fibers (Firts stage)             #
            # ================================================================#
            ii = 0
            if self.fiber_positions.shape[0] < self.num_fibers_max:
                # for every point, stirring is needed!
                while ii < self.fiber_positions.shape[0]:
                    (
                        self.fiber_min_dis_vector,
                        min_index,
                        min_dis,
                    ) = self.min_dis_index(
                        self.fiber_positions[ii, 0:2],
                        self.fiber_positions.copy(),
                        self.fiber_min_dis_vector,
                        ii,
                        self.num_cycle,
                    )

                    new_fiber_temp = self.generate_first_heuristic_fibers(
                        ref_point=self.fiber_positions[min_index, 0:3].copy(),
                        fiber_temp=self.fiber_positions[ii, 0:3].copy(),
                        dist_factor=self.dist_min_factor,
                    )
                    new_fiber = self.new_positions(
                        x_center=new_fiber_temp[0, 0],
                        y_center=new_fiber_temp[0, 1],
                        radius=new_fiber_temp[0, 2],
                        length=self.length,
                        width=self.width,
                    )
                    overlap_status = self.overlap_check(
                        new_fiber=new_fiber,
                        fiber_pos=self.fiber_positions.copy(),
                        dist_factor=self.dist_min_factor,
                        stage="step_two",
                        fiber_index=ii,
                    )
                    # check: if the new fibers(cause it maybe more than
                    # 1 fiber centers) will overlap with the
                    # remaining ones or not
                    if overlap_status == 0:
                        ii = self._update_fiber_position(
                            new_fiber=new_fiber, iter=ii
                        )
                    else:
                        ii = ii + int(self.fiber_positions[ii, 3])

                    del new_fiber, new_fiber_temp
            # end of one cycle
            self.num_cycle = self.num_cycle + 1

    def _update_fiber_position(self, new_fiber: np.ndarray, iter: int) -> int:
        """update the fiber position

        Parameters
        ----------
        new_fiber : np.ndarray
            the generated new fiber
        iter : int
            determine the nex fiber should be analysis

        Returns
        -------
        iter : int
            he updated number of index
        """
        # check the location compatibility
        if new_fiber[0, 3] != new_fiber.shape[0]:
            raise Exception("fiber number is wrong \n")

        if self.fiber_positions[iter, 3] == 1:
            # the original fiber is a full circle
            if new_fiber[0, 3] == 1:
                # keep one point after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions, (iter), axis=0
                )
                self.fiber_positions = np.insert(
                    self.fiber_positions, (iter), new_fiber, axis=0
                )
                iter = iter + 1
            elif new_fiber[0, 3] == 2:
                # split into two hal after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions, (iter), axis=0
                )
                # delete the original ii point first
                self.fiber_positions = np.insert(
                    self.fiber_positions, (iter), new_fiber, axis=0
                )
                # add two point at iith location
                iter = iter + 2
            elif new_fiber[0, 3] == 4:
                # split into foul one-fouth circles after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions, (iter), axis=0
                )
                # delete the original ii point first
                self.fiber_positions = np.insert(
                    self.fiber_positions,
                    (iter),
                    new_fiber,
                    axis=0,
                )
                # add four point at iith location
                iter = iter + 4
            else:
                raise KeyError("The new splits of fiber is wrong!!! \n")

        elif self.fiber_positions[iter, 3] == 2:
            if new_fiber[0, 3] == 2:
                # keep two half points after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions, (iter, iter + 1), axis=0
                )
                self.fiber_positions = np.insert(
                    self.fiber_positions, (iter), new_fiber, axis=0
                )
                iter = iter + 2
            elif new_fiber[0, 3] == 1:
                # reduce to one circle after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions, (iter, iter + 1), axis=0
                )
                # delete the ii+1 half fiber
                self.fiber_positions = np.insert(
                    self.fiber_positions, (iter), new_fiber, axis=0
                )
                # add four point at iith location
                iter = iter + 1
            elif new_fiber[0, 3] == 4:
                # reduce to one circle after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions, (iter, iter + 1), axis=0
                )
                # delete the ii+1 half fiber
                self.fiber_positions = np.insert(
                    self.fiber_positions,
                    (iter),
                    new_fiber,
                    axis=0,
                )
                # add four point at iith location
                iter = iter + 4
            else:
                print("The new splits of fiber is wrong!!! \n")

        elif self.fiber_positions[iter, 3] == 4:
            if new_fiber[0, 3] == 4:
                # reduce to one circle after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions,
                    (iter, iter + 1, iter + 2, iter + 3),
                    axis=0,
                )
                self.fiber_positions = np.insert(
                    self.fiber_positions, (iter), new_fiber, axis=0
                )
                iter = iter + 4
            elif new_fiber[0, 3] == 2:
                # reduce to one circle after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions,
                    (iter, iter + 1, iter + 2, iter + 3),
                    axis=0,
                )
                # delete the original ii point first
                self.fiber_positions = np.insert(
                    self.fiber_positions, (iter), new_fiber, axis=0
                )
                # add two point at iith location
                iter = iter + 2
            elif new_fiber[0, 3] == 1:
                # reduce to one circle after stirring
                self.fiber_positions = np.delete(
                    self.fiber_positions,
                    (iter, iter + 1, iter + 2, iter + 3),
                    axis=0,
                )
                # delete the original ii point first
                self.fiber_positions = np.insert(
                    self.fiber_positions, (iter), new_fiber, axis=0
                )
                # add two point at iith location
                iter = iter + 1
            else:
                raise KeyError("The new splits of fiber is wrong!!! \n")

        else:
            raise KeyError("Can not match the overlap condition  !!! \n")

        return iter

    def crate_rgmsh(self, num_discrete: int = 10) -> np.ndarray:
        """create rgmsh numpy array for crate

        Parameters
        ----------
        num_discrete : int, optional
            number of discrete partition, by default 10

        Returns
        -------
        np.ndarray
            a 2d numpy array that contains the micro-structure information
        """

        self.rgmsh = np.zeros((num_discrete, num_discrete))
        grid_len = self.length / num_discrete
        grid_wid = self.width / num_discrete
        radius = self.fiber_positions[:, 2].reshape(-1, 1)
        for ii in range(num_discrete):
            for jj in range(num_discrete):
                loc_temp = np.array(
                    [
                        [
                            self.length / (2 * num_discrete) + ii * grid_len,
                            self.width / (2 * num_discrete) + jj * grid_wid,
                        ]
                    ]
                )
                # distance measure
                points_dis_temp = distance_matrix(
                    self.fiber_positions[:, 0:2],
                    loc_temp,
                )

                if (points_dis_temp - radius).min() < 0:
                    self.rgmsh[ii, jj] = 1

        return self.rgmsh.T

    def rgmsh_plot(self) -> None:

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(
            self.rgmsh.T,
            origin="lower",
            interpolation="None",
            cmap="summer_r",
        )
        fig.colorbar(im)
        plt.title(f"$V_f$ = {self.vol_frac*100:.2f}")
        plt.show()

    @staticmethod
    def fiber_volume(radius: float) -> float:
        """calculate the fiber volume of the current fiber

        Parameters
        ----------
        radius : float
            radius

        Returns
        -------
        vol:float
            volume of current fiber(disk)
        """
        return np.pi * radius**2

    @staticmethod
    def generate_random_fibers(
        len_start: float,
        len_end: float,
        wid_start: float,
        wid_end: float,
        radius_mu: float,
        radius_std: float,
    ) -> np.ndarray:
        """generate random fibers with different radiis

        Parameters
        ----------
        len_start : float
            the start location of length
        len_end : float
            the end location of length
        wid_start : float
            the start location of width
        wid_end : float
            the end location of width
        radius_mu : float
            mean of radius
        radius_std : float
            standard deviation of radius

        Returns
        -------
        np.ndarray
            location information of generated fiber
        """

        x = np.random.uniform(len_start, len_end, 1)
        y = np.random.uniform(wid_start, wid_end, 1)
        r = np.random.normal(radius_mu, radius_std, 1)
        while r <= 0:
            r = np.random.normal(radius_mu, radius_std, 1)
        fiber = np.array([x, y, r])
        return fiber

    @staticmethod
    def new_positions(
        x_center: float,
        y_center: float,
        radius: float,
        width: float,
        length: float,
    ) -> np.ndarray:
        """This is the function used to determine the locations of disks
        considering the boundary of RVE. To be specific, the disk should
        be divided into two parts or four parts if it is located in the
        outskirt area

        Parameters
        ----------
        x_center : float
            new center of X axis
        y_center : float
            new center of Y axis
        radius : float
            radius of the disk
        width : float
            width of RVE
        length : float
            length of RVE

        Returns
        -------
        np.ndarray
            XC, YC, split  which is the new locations of this fiber
            (because in some locations, the circles need to be split)
        ####################################
        #     #          2_2       #       #
        # 4_1 #                    #4_2    #
        ####################################
        #     #                    #       #
        # 2_3 #          1         #   2_4 #
        #     #                    #       #
        #     #                    #       #
        ####################################
        # 4_3 #         2_1        # 4_4   #
        #     #                    #       #
        ####################################

        """

        new_fiber = np.zeros((1, 4))
        if (
            radius <= x_center <= length - radius
            and radius <= y_center <= width - radius
        ):
            # locate in center region and split = 1
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 1

        elif length - radius > x_center > radius > y_center:
            # location 2_1
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 2
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center, y_center + width, radius, 2]).reshape(
                        (1, 4)
                    ),
                )
            )

        elif radius < x_center < length - radius and y_center > width - radius:
            # location 2_2
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 2
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center, y_center - width, radius, 2]).reshape(
                        (1, 4)
                    ),
                )
            )

        elif width - radius > y_center > radius > x_center:
            # location 2_3
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 2
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center + length, y_center, radius, 2]).reshape(
                        (1, 4)
                    ),
                )
            )

        elif radius < y_center < width - radius and x_center > length - radius:
            # location 2_4
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 2
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center - length, y_center, radius, 2]).reshape(
                        (1, 4)
                    ),
                )
            )

        elif x_center < radius and y_center > width - radius:
            # location 4_1
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 4
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center + length, y_center, radius, 4]).reshape(
                        (1, 4)
                    ),
                )
            )
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array(
                        [x_center + length, y_center - width, radius, 4]
                    ).reshape((1, 4)),
                )
            )
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center, y_center - width, radius, 4]).reshape(
                        (1, 4)
                    ),
                )
            )

        elif x_center > length - radius and y_center > width - radius:
            # location 4_2
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 4
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center - length, y_center, radius, 4]).reshape(
                        (1, 4)
                    ),
                )
            )
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array(
                        [x_center - length, y_center - width, radius, 4]
                    ).reshape((1, 4)),
                )
            )
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center, y_center - width, radius, 4]).reshape(
                        (1, 4)
                    ),
                )
            )

        elif x_center < radius and y_center < radius:
            # location 4_3
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 4
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center + length, y_center, radius, 4]).reshape(
                        (1, 4)
                    ),
                )
            )
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array(
                        [x_center + length, y_center + width, radius, 4]
                    ).reshape((1, 4)),
                )
            )
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center, y_center + width, radius, 4]).reshape(
                        (1, 4)
                    ),
                )
            )

        elif x_center > length - radius and y_center < radius:
            # location 4_4
            new_fiber[0, 0] = x_center
            new_fiber[0, 1] = y_center
            new_fiber[0, 2] = radius
            new_fiber[0, 3] = 4
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center - length, y_center, radius, 4]).reshape(
                        (1, 4)
                    ),
                )
            )
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array(
                        [x_center - length, y_center + width, radius, 4]
                    ).reshape((1, 4)),
                )
            )
            new_fiber = np.vstack(
                (
                    new_fiber,
                    np.array([x_center, y_center + width, radius, 4]).reshape(
                        (1, 4)
                    ),
                )
            )

        else:
            raise Exception(
                "The location of the original point was wrong!!! \n"
            )

        return new_fiber

    @staticmethod
    def overlap_check(
        new_fiber: np.ndarray,
        fiber_pos: np.ndarray,
        dist_factor: float,
        fiber_index: int = 0,
        stage: str = "step_one",
    ) -> int:
        """Check if the new positions will overlap with the original one
        or not ?
        Parameters

        Parameters
        ----------
        new_fiber : np.ndarray
            he locations of the new point(in some cases: several
            positions will be generated for one point because of
            the splitting )
        fiber_pos : np.ndarray
            location of the original points
        dist_min : float
            the allowed minimum distance between disks
        index : int, optional
            the index of the checking fibers, which should not be 0
            when excuting stage two and three
        stage : str, optional
            the stage of the algorithm, step_one, step_two,step_three
            , by default "step_one"

        Returns
        -------
        int
            overlap status, 0 is non-overlap and 1 means there is
            overlap
        """
        fiber_pos = fiber_pos.copy()

        if stage == "step_one":
            min_dis_threhold = dist_factor * (
                new_fiber[0, 2] + fiber_pos[:, 2]
            ).reshape((-1, 1))
            points_dis_temp = distance_matrix(
                fiber_pos[:, 0:2], new_fiber[:, 0:2]
            )
            points_dis = np.min(points_dis_temp, 1, keepdims=True)
            min_dis = points_dis - min_dis_threhold

        elif stage == "step_two":
            # calculate the minmum distance threshold
            min_dis_threhold = dist_factor * (
                new_fiber[0, 2] + fiber_pos[:, 2]
            ).reshape((-1, 1))
            points_dis_temp = distance_matrix(
                fiber_pos[:, 0:2], new_fiber[:, 0:2]
            )
            points_dis_temp[
                fiber_index : fiber_index + int(fiber_pos[fiber_index, 3]), :
            ] = math.inf
            points_dis = np.min(points_dis_temp, 1, keepdims=True)
            min_dis = points_dis - min_dis_threhold

        else:
            raise ValueError(" Not defined stage \n")

        if min_dis.min() <= 0:
            status = 1
        else:
            status = 0

        return status

    @staticmethod
    def min_dis_index(
        temp_fiber: np.ndarray,
        fiber_pos: np.ndarray,
        fiber_min_dis_vector: np.ndarray,
        ii: int,
        cycle: int,
    ) -> list[np.ndarray, int, float]:
        """This function is used to identify the index of closest fiber
        of every fiber, which is very import for the first heuristic
        stirring to get more space placing the new disks.

        Parameters
        ----------
        temp_fiber : np.ndarray
            the fiber been processed
        fiber_pos : np.ndarray
            fiber position
        fiber_min_dis_vector : np.ndarray
            the first column is the index of the closed point,
            the second column contains the minimum distance between
            those two points
        ii : int
            the index of the being processed point
        cycle : int
            the cycle of the algorithm. At different cycles, the
            criteria of identifying the closed point are  different.

        Returns
        -------
        fiber_min_dis_vector: np.ndarray
            The updated minimum distance array
        two: int
            The index of the minimum distance point
        three : float
            The minimum distance to the minimum distance point

        """
        fiber_pos = fiber_pos.copy()

        # pre-process the data : find out the same row data and delete it
        temp_fiber = temp_fiber.reshape((1, 2))
        points_dis = distance_matrix(fiber_pos[:, 0:2], temp_fiber)
        points_dis[points_dis == 0] = math.inf
        if cycle == 0:
            min_dis = points_dis.min()
            min_index = np.where(points_dis == min_dis)[0]
            fiber_min_dis_vector[ii, cycle, 0] = min_index
            fiber_min_dis_vector[ii, cycle, 1] = min_dis
        elif cycle == 1:
            index_pre = int(fiber_min_dis_vector[ii, cycle - 1, 0])
            if index_pre < points_dis.shape[0]:
                points_dis[index_pre, :] = math.inf
            # identify the minimum index
            min_dis = points_dis.min()
            min_index = np.where(points_dis == min_dis)[0]
            fiber_min_dis_vector[ii, cycle, 0] = min_index
            fiber_min_dis_vector[ii, cycle, 1] = min_dis
        else:

            index_pre = int(fiber_min_dis_vector[ii, cycle - 1, 0])
            index_pre_pre = int(fiber_min_dis_vector[ii, cycle - 2, 0])
            if (
                index_pre < points_dis.shape[0]
                and index_pre_pre < points_dis.shape[0]
            ):
                points_dis[index_pre, :] = math.inf
                points_dis[index_pre_pre, :] = math.inf
            # identify the minimum index
            min_dis = points_dis.min()
            min_index = np.where(points_dis == min_dis)[0]
            fiber_min_dis_vector[ii, cycle, 0] = min_index
            fiber_min_dis_vector[ii, cycle, 1] = min_dis

        return fiber_min_dis_vector, min_index, min_dis

    @staticmethod
    def generate_first_heuristic_fibers(
        ref_point: np.ndarray, fiber_temp: np.ndarray, dist_factor: float
    ) -> np.ndarray:
        """Move fiber to its reference point

        Parameters
        ----------
        ref_point : np.ndarray
            Reference point that the fiber should move to
        fiber_temp : np.ndarray
            The considering fiber
        dist_factor : float
            the minimum distance factor between two fibers

        Returns
        -------
        np.ndarray
            The updated location of the considering fiber
        """

        fiber_temp = fiber_temp.reshape((1, 3))
        ref_point = ref_point.reshape((1, 3))
        # generate the random factor for fiber stirring
        delta = np.random.uniform(0, 1, 1)
        dist_min = dist_factor * (fiber_temp[0, 2] + ref_point[0, 2])
        fiber_loc = fiber_temp[0, 0:2].reshape((1, 2)).copy()
        ref_loc = ref_point[0, 0:2].reshape((1, 2)).copy()
        # maximum length of movement
        k = 1 - dist_min / distance_matrix(ref_loc, fiber_loc)
        fiber_temp[0, 0:2] = fiber_loc + delta * k * (ref_loc - fiber_loc)

        return fiber_temp
