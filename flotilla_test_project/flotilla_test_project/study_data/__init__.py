__author__ = 'lovci'

import os, sys
study_data_dir = os.path.dirname(__file__)
with open(os.path.join(study_data_dir, "README"), 'r') as f:
    for line in f.readlines():
        if not line.startswith("#"):
            sys.stderr.write(line)