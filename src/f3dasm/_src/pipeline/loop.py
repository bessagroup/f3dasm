"""Loop construct for repeating pipeline steps."""


#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Local
if TYPE_CHECKING:
    from .pipeline import Step

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


@dataclass
class Loop:
    """A group of steps that are repeated multiple times.

    Use ``Loop`` inside a :class:`Pipeline`'s ``steps`` list to
    express iterative workflows (e.g. train → evaluate loops).

    In SLURM mode, a self-resubmitting orchestrator script is
    generated so that only one iteration's jobs are queued at a
    time, preventing cluster job-count limits from being exceeded.

    Parameters
    ----------
    n_iterations : int
        Number of times to repeat the inner steps.
    steps : list[Step]
        The steps to repeat each iteration.

    Examples
    --------
    ::

        Pipeline(
            name="online_rl",
            steps=[
                Step("create", block=create_block),
                Loop(n_iterations=10, steps=[
                    Step("run", block=generator,
                         parallel=True),
                    Step("post", block=update_block),
                ]),
            ],
        )
    """

    n_iterations: int = 1
    steps: list[Step] = field(default_factory=list)
