"""Post-processing block that computes the scalar objective column.

The ``SupercompressibleObjective`` block reads ``coilable`` and
``sigma_crit`` from each row in an ``ExperimentData`` object and writes
a scalar ``objective`` output column according to the rule defined in
``docs/specs/supercompressible-baseline.md``:

- If ``coilable == 1``: ``objective = sigma_crit``  (maximise)
- Otherwise (``coilable`` in {0, 2}): ``objective = penalty``

The block is idempotent — running it twice on the same data overwrites
the previously written ``objective`` values with identical values, so
the net effect is unchanged.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Local
from f3dasm import Block, ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# =============================================================================
# Helper — scalar objective rule
# =============================================================================


def _compute_objective(coilable: float, sigma_crit: float,
                       penalty: float) -> float:
    """Return the scalar objective for one design row.

    Parameters
    ----------
    coilable : float
        Coilability class: 0 = wrong buckling mode,
        1 = reversibly coilable, 2 = coils but fractures.
    sigma_crit : float
        Critical buckling stress in kPa.
    penalty : float
        Penalty value applied when ``coilable != 1``.

    Returns
    -------
    float
        Scalar objective value.
    """
    # Only reversibly-coilable designs receive a positive reward.
    if int(coilable) == 1:
        return float(sigma_crit)
    # coilable in {0, 2}: apply a uniform large negative penalty so that
    # the optimizer always prefers any coilable=1 design.
    return penalty


# =============================================================================
# Block subclass
# =============================================================================


class SupercompressibleObjective(Block):
    """Compute the scalar ``objective`` output column for the supercompressible
    study.

    Iterates over every row in ``data`` and writes
    ``_output_data["objective"]`` using the coilable/sigma_crit rule from
    the baseline spec.  Rows where ``coilable`` or ``sigma_crit`` are
    ``None`` (e.g. Stage-1-only evaluations) receive the penalty value.

    Parameters
    ----------
    penalty : float, optional
        Penalty applied when ``coilable != 1``.  Default is ``-1e6``.
    """

    def __init__(self, penalty: float = -1e6) -> None:
        """Initialise the block.

        Parameters
        ----------
        penalty : float, optional
            Penalty applied to non-coilable designs.
        """
        self._penalty = penalty

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """Write the ``objective`` column into every row of ``data``.

        Parameters
        ----------
        data : ExperimentData
            The dataset to process.  Modified in place.
        **kwargs
            Accepted for signature compatibility; not used.

        Returns
        -------
        ExperimentData
            The same ``data`` object with ``objective`` populated.
        """
        # Ensure the objective output column exists on the domain so that
        # f3dasm's schema-bound ExperimentData will accept the writes.
        data.domain.add_output("objective", to_disk=False, exist_ok=True)

        for sample in data.data.values():
            coilable = sample._output_data.get("coilable")
            sigma_crit = sample._output_data.get("sigma_crit")

            # Handle missing data: apply penalty if either field is absent.
            if coilable is None or sigma_crit is None:
                sample._output_data["objective"] = self._penalty
                continue

            sample._output_data["objective"] = _compute_objective(
                coilable=coilable,
                sigma_crit=sigma_crit,
                penalty=self._penalty,
            )

        return data
