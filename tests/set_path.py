import os
import sys

cur_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
if par_dir not in sys.path:
    sys.path.append(par_dir)