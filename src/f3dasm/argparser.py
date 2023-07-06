import argparse

parser = argparse.ArgumentParser()

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Option with ++ prefix
parser.add_argument('--jobid', type=int, help='The PBS job ID for HPC', default=None)

# Parse the command-line arguments
args = parser.parse_args()

# Access the value of the --hpc-jobid flag
HPC_JOBID = args.jobid
