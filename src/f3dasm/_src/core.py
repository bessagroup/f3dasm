"""
This module contains the core blocks and protocols for the f3dasm package.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import traceback
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Literal, Optional,
                    Protocol, Tuple, Type)

# Third-party
import numpy as np
import pandas as pd
from filelock import FileLock
from pathos.helpers import mp

from ._io import EXPERIMENTDATA_SUBFOLDER, LOCK_FILENAME, MAX_TRIES
# Local
from .logger import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


class Block(ABC):
    """
    Abstract base class representing an operation in the data-driven process
    """

    def arm(self, data: ExperimentData) -> None:
        """
        Prepare the block with a given ExperimentData.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to be used by the block.

        Notes
        -----
        This method can be inherited by a subclasses to prepare the block
        with the given experiment data. It is not required to implement this
        method in the subclass.
        """
        pass

    @abstractmethod
    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """
        Execute the block's operation on the ExperimentData.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to process.
        **kwargs : dict
            Additional keyword arguments for the operation.

        Returns
        -------
        ExperimentData
            The processed experiment data.
        """
        pass


class LoopBlock(Block):
    def __init__(self, blocks: Block | Iterable[Block], n_loops: int):
        """
        Initialize a LoopBlock instance.

        Parameters
        ----------
        blocks : Block or Iterable[Block]
            The block or blocks to loop over.
        n_loops : int
            The number of loops to perform.
        """
        if isinstance(blocks, Block):
            blocks = [blocks]

        self.blocks = blocks
        self.n_loops = n_loops

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """
        Execute the looped blocks on the ExperimentData.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to process.
        **kwargs : dict
            Additional keyword arguments for the blocks.

        Returns
        -------
        ExperimentData
            The processed experiment data after looping.
        """
        for _ in range(self.n_loops):
            for block in self.blocks:
                block.arm(data)
                data = block.call(data=data, **kwargs)

        return data


def loop(blocks: Block | Iterable[Block], n_loops: int) -> Block:
    """
    Create a loop to execute blocks multiple times.

    Parameters
    ----------
    blocks : Block or Iterable[Block]
        The block or blocks to loop over.
    n_loops : int
        The number of loops to perform.

    Returns
    -------
    Block
        An new Block instance that loops over the given blocks.
    """
    return LoopBlock(blocks=blocks, n_loops=n_loops)

# =============================================================================


class Domain(Protocol):
    ...


class ExperimentSample(Protocol):
    def mark(self,
             status: Literal['open', 'in_progress', 'finished', 'error']):
        ...

    def store(self,
              name: str, object: Any, to_disk: bool):
        ...

    def store_experimentsample_references(self, idx: int) -> None:
        ...

    @property
    def input_data(self) -> Dict[str, Any]:
        ...

    @property
    def output_data(self) -> Dict[str, Any]:
        ...


class ExperimentData(Protocol):
    def __init__(self, domain: Domain, input_data: np.ndarray,
                 output_data: np.ndarray):
        ...

    @property
    def domain(self) -> Domain:
        ...

    @property
    def project_dir(self) -> Path:
        ...

    @property
    def index(self) -> pd.Index:
        ...

    @classmethod
    def from_sampling(cls, domain: Domain, sampler: Block,
                      n_samples: int, seed: int) -> ExperimentData:
        ...

    def access_file(self, operation: Callable) -> Callable:
        ...

    def get_open_job(self) -> Tuple[int, ExperimentSample]:
        ...

    def store_experimentsample(self,
                               experiment_sample: ExperimentSample, idx: int):
        ...

    def from_file(self, project_dir: str, wait_for_creation: bool,
                  max_tries: int) -> ExperimentData:
        ...

    def store(self) -> None:
        ...

    def mark(self, indices: int | Iterable[int],
             status: Literal['open', 'in_progress', 'finished', 'error']):
        ...

    def remove_lockfile(self) -> None:
        ...

    def sample(self, sampler: Block, **kwargs):
        ...

    def evaluate(self, data_generator: DataGenerator, mode:
                 str, output_names: Optional[List[str]] = None, **kwargs):
        ...

    def get_n_best_output(self, n_samples: int) -> ExperimentData:
        ...

    def to_numpy() -> Tuple[np.ndarray, np.ndarray]:
        ...

    def select(self, indices: int | slice | Iterable[int]) -> ExperimentData:
        ...

    def get_experiment_sample(self, id: int) -> ExperimentData:
        ...

    def remove_rows_bottom(self, number_of_rows: int):
        ...

    def add_experiments(self, experiment_sample: ExperimentData):
        ...

    def _overwrite_experiments(self, experiment_sample: ExperimentData,
                               indices: pd.Index, add_if_not_exist: bool):
        ...

    def _reset_index(self):
        ...


# =============================================================================


class DataGenerator(Block):
    """Base class for a data generator"""

    def call(self, data: ExperimentData | str,
             mode: str = 'sequential', **kwargs
             ) -> ExperimentData:
        """
        Evaluate the data generator.

        Parameters
        ----------
        mode : str, optional
            The mode of evaluation, by default 'sequential'
        kwargs : dict
            The keyword arguments to pass to the pre_process, execute and
            post_process

        Returns
        -------
        ExperimentData
            The processed data
        """
        if mode == 'sequential':
            return self._evaluate_sequential(data=data, **kwargs)
        elif mode == 'parallel':
            return self._evaluate_multiprocessing(data=data, **kwargs)
        elif mode.lower() == "cluster":
            return self._evaluate_cluster(data=data, **kwargs)
        else:
            raise ValueError(f"Invalid parallelization mode specified: {mode}")

    # =========================================================================

    def _evaluate_sequential(self, data: ExperimentData, **kwargs
                             ) -> ExperimentData:
        """Run the operation sequentially

        Parameters
        ----------
        kwargs : dict
            Any keyword arguments that need to be supplied to the function

        Raises
        ------
        NoOpenJobsError
            Raised when there are no open jobs left
        """

        while True:
            job_number, experiment_sample = data.get_open_job()
            if job_number is None:
                logger.debug("No Open jobs left!")
                break

            try:
                experiment_sample: ExperimentSample = self.execute(
                    experiment_sample=experiment_sample, **kwargs)

                experiment_sample.store_experimentsample_references(
                    idx=job_number)

                experiment_sample.mark('finished')

            except Exception:
                error_msg = f"Error in experiment_sample {job_number}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                experiment_sample.mark('error')
                continue

            finally:
                data.store_experimentsample(
                    idx=job_number, experiment_sample=experiment_sample,
                )
        return data

        # while True:

        #     job_number, experiment_sample = data.get_open_job()
        #     logger.debug(
        #         f"Accessed experiment_sample \
        #                 {job_number}")
        #     if job_number is None:
        #         logger.debug("No Open Jobs left")
        #         break

        #     try:

        #         # # If kwargs is empty dict
        #         # if not kwargs:
        #         #     logger.debug(
        #         #         f"Running experiment_sample "
        #         #         f"{job_number}")
        #         # else:
        #         #     logger.debug(
        #         #         f"Running experiment_sample "
        #         #         f"{job_number} with kwargs {kwargs}")
        #         experiment_sample: ExperimentSample = self.execute(
        #             experiment_sample=experiment_sample, **kwargs)

        #         # _experiment_sample = self._run(
        #         #     experiment_sample, **kwargs)  # no *args!

        #         data.store_experimentsample(
        #             experiment_sample=experiment_sample,
        #             idx=job_number)
        #     except Exception as e:
        #         error_msg = f"Error in experiment_sample \
        #              {job_number}: {e}"
        #         error_traceback = traceback.format_exc()
        #         logger.error(f"{error_msg}\n{error_traceback}")
        #         data.mark(indices=job_number, status='error')

        # return data

    def _evaluate_multiprocessing(
        self, data: ExperimentData,
            nodes: int = mp.cpu_count(), **kwargs) -> ExperimentData:
        options = []

        while True:
            job_number, experiment_sample = data.get_open_job()
            if job_number is None:
                break
            options.append(
                ({'experiment_sample': experiment_sample,
                  '_job_number': job_number, **kwargs},))

        def f(options: Dict[str, Any]) -> Tuple[int, ExperimentSample, int]:
            job_number = options.pop('_job_number')
            try:

                logger.debug(
                    f"Running experiment_sample "
                    f"{job_number}")

                experiment_sample: ExperimentSample = self.execute(**options)
                experiment_sample.store_experimentsample_references(
                    idx=job_number)
                # exit_code = 0
                experiment_sample.mark('finished')

            except Exception:
                error_msg = f"Error in experiment_sample {job_number}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                experiment_sample.mark('error')
                # exit_code = 1

            finally:
                return (job_number, experiment_sample)

            # except Exception as e:
            #     error_msg = f"Error in experiment_sample \
            #          {options['_job_number']}: {e}"
            #     error_traceback = traceback.format_exc()
            #     logger.error(f"{error_msg}\n{error_traceback}")
            #     return (options['_job_number'],
            #             options['experiment_sample'], 1)

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            _experiment_samples: List[
                Tuple[int, ExperimentSample, int]] = pool.starmap(f, options)

        for job_number, experiment_sample in _experiment_samples:
            # if exit_code == 0:
            data.store_experimentsample(
                experiment_sample=experiment_sample,
                idx=job_number)
            # else:
            #     data.mark(indices=job_number, status='error')

        return data

    # def _evaluate_cluster(
    #         self, data: ExperimentData, **kwargs) -> ExperimentData:
    #     """Run the operation on the cluster

    #     Parameters
    #     ----------
    #     operation : ExperimentSampleCallable
    #         function execution for every entry in the ExperimentData object
    #     kwargs : dict
    #         Any keyword arguments that need to be supplied to the function

    #     Raises
    #     ------
    #     NoOpenJobsError
    #         Raised when there are no open jobs left
    #     """
    #     data = type(data).from_file(data.project_dir)

    #     get_open_job = data.access_file(type(data).get_open_job)
    #     store_experiment_sample = data.access_file(
    #         type(data).store_experimentsample)
    #     mark = data.access_file(type(data).mark)

    #     while True:
    #         job_number, experiment_sample = get_open_job()
    #         if job_number is None:
    #             logger.debug("No Open jobs left!")
    #             break

    #         try:
    #             _experiment_sample = self._run(
    #                 experiment_sample, **kwargs)
    #             store_experiment_sample(experiment_sample=_experiment_sample,
    #                                     id=job_number)
    #         except Exception:
    #             # n = experiment_sample.job_number
    #             error_msg = f"Error in experiment_sample {job_number}: "
    #             error_traceback = traceback.format_exc()
    #             logger.error(f"{error_msg}\n{error_traceback}")
    #             mark(indices=job_number, status='error')
    #             continue

    #     data = type(data).from_file(data.project_dir)

    #     # Remove the lockfile from disk
    #     data.remove_lockfile()
    #     return data

    def _evaluate_cluster(
        self, data: ExperimentData,
            wait_for_creation: bool = False,
            max_tries: int = MAX_TRIES, **kwargs
    ) -> None:

        # Creat lockfile
        lockfile = FileLock(
            (data.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
             ).with_suffix('.lock'))

        cluster_get_open_job = partial(
            get_open_job, experiment_data_type=type(data),
            project_dir=data.project_dir,
            wait_for_creation=wait_for_creation, max_tries=max_tries,
            lockfile=lockfile)
        cluster_store_experiment_sample = partial(
            store_experiment_sample, experiment_data_type=type(data),
            project_dir=data.project_dir,
            wait_for_creation=wait_for_creation, max_tries=max_tries,
            lockfile=lockfile)

        while True:
            job_number, experiment_sample = cluster_get_open_job()
            if job_number is None:
                logger.debug("No Open jobs left!")
                break

            try:
                experiment_sample: ExperimentSample = self.execute(
                    experiment_sample=experiment_sample, **kwargs)

                experiment_sample.store_experimentsample_references(
                    idx=job_number)

                experiment_sample.mark('finished')

            except Exception:
                error_msg = f"Error in experiment_sample {job_number}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                experiment_sample.mark('error')
                continue

            finally:
                cluster_store_experiment_sample(
                    idx=job_number, experiment_sample=experiment_sample)

        # Remove lockfile
        (data.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
         ).with_suffix('.lock').unlink(missing_ok=True)

        # =========================================================================

    @abstractmethod
    def execute(self, experiment_sample: ExperimentSample,
                **kwargs) -> ExperimentSample:
        """Interface function that handles the execution of the data generator

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The experiment sample to run the data generator on
        kwargs : dict
            The optional keyword arguments to pass to the function

        Returns
        -------
        ExperimentSample
            The experiment sample with the response of the data generator
            saved in the experiment_sample

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the user
        """
        ...

    # def _run(
    #         self, experiment_sample: ExperimentSample,
    #         **kwargs) -> ExperimentSample:
    #     """
    #     Run the data generator.

    #     The function also caches the experiment_sample in the data generator.
    #     This allows the user to access the experiment_sample in the
    #     execute function as a class variable
    #     called self.experiment_sample.

    #     Parameters
    #     ----------
    #     ExperimentSample : ExperimentSample
    #         The design to run the data generator on

    #     kwargs : dict
    #         The keyword arguments to pass to the pre_process, execute \
    #         and post_process

    #     Returns
    #     -------
    #     ExperimentSample
    #         Processed design with the response of the data generator \
    #         saved in the experiment_sample
    #     """
    #     self.experiment_sample = experiment_sample

    #     self.experiment_sample.mark('in_progress')

    #     self.execute(**kwargs)

    #     self.experiment_sample.mark('finished')

    #     return self.experiment_sample

# =============================================================================

# lockfile = FileLock(
#     (project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
#      ).with_suffix('.lock'))


def get_open_job(experiment_data_type: Type[ExperimentData],
                 project_dir: Path, lockfile: FileLock,
                 wait_for_creation: bool, max_tries: int,
                 ) -> Tuple[int, ExperimentSample]:

    with lockfile:
        data = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        idx, es = data.get_open_job()

        data.store(project_dir)

    return idx, es


def store_experiment_sample(
    experiment_data_type: Type[ExperimentData],
        project_dir: Path, lockfile: FileLock, wait_for_creation: bool,
        max_tries: int, idx: int, experiment_sample: ExperimentSample) -> None:

    with lockfile:
        data: ExperimentData = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        data.store_experimentsample(experiment_sample=experiment_sample,
                                    idx=idx)
        data.store(project_dir)
