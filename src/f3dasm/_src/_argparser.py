#                                                                       Modules
# =============================================================================

# Standard
import argparse

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise RuntimeError(message)


# if all(("ipykernel_launcher" not in sys.argv[0],
#  os.environ.get("PYTEST_RUN") != "True")):
# Create an ArgumentParser object
parser = ArgumentParser()

# Option with ++ prefix
parser.add_argument('--jobid', type=int,
                    help='The PBS job ID for HPC', default=None)

# Parse the command-line arguments
try:
    args = parser.parse_args()
    # Access the value of the --hpc-jobid flag
    HPC_JOBID = args.jobid
except RuntimeError:
    HPC_JOBID = None
