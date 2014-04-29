__author__ = 'lovci'


#import socket
#if "compute" in socket.gethostname():
#    flotilla_data = "/nas3/lovci/projects/singleCell/notebook2/distribute/"
#elif "tscc" in socket.gethostname():
#    flotilla_data = "/home/mlovci/scratch/singleCell/regressions/distribute/"
#else:
#    raise Exception

import os

def data_dir():
    """
    project-agnostic data_dir
    """
    return os.path.join(os.path.dirname(__file__), '../data')

flotilla_data = data_dir()
