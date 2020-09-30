'''
Created on 2020-09-08 16:30:52
Last modified on 2020-09-08 16:44:16

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


def get_wait_time_from_log(job_name):
    with open('{}.log'.format(job_name), 'r') as file:
        lines = file.readlines()

    waited = False
    for line in lines:
        if 'Queued for QXT' in line:
            waited = True
            break
    if waited:
        for line in lines[::-1]:
            if 'Total time in queue' in line:
                wait_time = [int(spl) for spl in line.split() if spl.isdigit()][0]
                break
    else:
        wait_time = 0

    return wait_time
