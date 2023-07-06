import argparse

# Argparse jobid if applicable


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add the --hpc-jobid flag with a default value
parser.add_argument("++jobid", type=int, default=None, help="The PBS job ID for HPC")

# Parse the command-line arguments
args = parser.parse_args()

# Access the value of the --hpc-jobid flag
HPC_JOBID = args.jobid
