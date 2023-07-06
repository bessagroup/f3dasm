#                                                                       Modules
# =============================================================================

# Standard
import argparse
import os
import sys

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

if all(("ipykernel_launcher" not in sys.argv[0], os.environ.get("PYTEST_RUN") != "True")):
    parser = argparse.ArgumentParser()

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Option with ++ prefix
    parser.add_argument('--jobid', type=int, help='The PBS job ID for HPC', default=None)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the value of the --hpc-jobid flag
    HPC_JOBID = args.jobid

else:
    HPC_JOBID = None
