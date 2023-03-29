import errno
import fcntl
import json
from time import sleep
from typing import Callable, Union

from ..design import ExperimentData


def access_file(sleeptime_sec: int = 1) -> Callable:
    def decorator_func(operation: Callable) -> Callable:
        def wrapper_func(self, *args, **kwargs) -> None:
            while True:
                try:
                    # Try to open the jobs file
                    with open(f"{self.filename}.json", 'a+') as file:
                        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                        # Load the jobs data to the object
                        self.create_jobs_from_dictionary(json.load(file))

                        # Do the operation
                        value = operation(self, *args, **kwargs)

                        # Write the data back
                        json.dump(self.get_jobs(), file)

                    # with open(f"{self.filename}.json", 'w') as file:

                    # # remove the lock
                    # fcntl.flock(file, fcntl.LOCK_UN)

                    break
                except IOError as e:
                    # the file is locked by another process
                    if e.errno == errno.EAGAIN:
                        print("The jobs file is currently locked by another process. Retrying in 1 second...")
                        sleep(sleeptime_sec)
                    else:
                        print(f"An unexpected IOError occurred: {e}")
                        break
                except Exception as e:
                    # handle any other exceptions
                    print(f"An unexpected error occurred: {e}")
                    break
            return value
        return wrapper_func
    return decorator_func


class Jobs:
    def __init__(self, filename: str):
        self.filename = filename

    def create_jobs_from_experimentdata(self, experimentdata: ExperimentData):
        self.jobs = {index: 'open' for index in range(experimentdata.get_number_of_datapoints())}

    def create_jobs_from_dictionary(self, dictionary: dict):

        # Convert str keys to int
        new_dict = {}
        for k, v in dictionary.items():
            new_dict[int(k)] = v

        self.jobs = new_dict

    def get_jobs(self) -> dict:
        return self.jobs

    def set_value(self, index: int, value: str):
        self.jobs[index] = value

    @access_file()
    def add_job(self, index: int):
        end = len(self.jobs)
        self.jobs[end] = 'open'

    @access_file()
    def get(self) -> Union[int, None]:
        for key, value in self.jobs.items():
            if value == 'open':
                return key

        # if no open job is present
        return None

    def __repr__(self):
        return self.jobs.__repr__()


def write_jobs(job: Jobs):
    with open(f"{job.filename}.json", 'w') as f:
        json.dump(job.get_jobs(), f)


# def read_jobs(name: str) -> Jobs:
#     with open(f"{name}.json") as f:
#         job_dict = json.load(f)
#     jobs = Jobs()

#     jobs.create_jobs_from_dictionary(job_dict)
#     return jobs
