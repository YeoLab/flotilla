__author__ = 'lovci'

"""

general utilities

"""

###http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish###
from functools import wraps
import errno
import os
import signal

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator
###http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish###

import os,sys,subprocess
from subprocess import PIPE
from ..project.notebook import notebook_dir
def serve_ipython(path=notebook_dir):

    c = subprocess.Popen(['ipython', 'notebook', '--script', '--notebook-dir', path], stdin=PIPE)
    try:
        c.wait()
    except KeyboardInterrupt:
        c.terminate()

def dict_to_str(dic):
        """join dictionary data into a string with that data"""
        return "_".join([k+ ":" + str(v) for (k,v) in dic.items()])
