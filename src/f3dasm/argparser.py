import argparse


class DoublePlusPrefixAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


parser = argparse.ArgumentParser()

# Argparse jobid if applicable


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Option with ++ prefix
parser.add_argument('++jobid', type=int, action=DoublePlusPrefixAction, help='The PBS job ID for HPC', default=None)

# Parse the command-line arguments
args = parser.parse_args()

# Access the value of the --hpc-jobid flag
HPC_JOBID = args.jobid
