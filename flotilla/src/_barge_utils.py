__author__ = 'lovci'

"""

general utilities

"""

from functools import wraps
import errno
import os
import signal
import sys
import subprocess

###http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish###


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

t
def serve_ipython():
    try:

        assert len(sys.argv) == 2
        path = sys.argv[1]
        assert os.path.exists(sys.argv[1])

    except:
        raise ValueError("specify a notebook directory as the first and only argument")

    c = subprocess.Popen(['ipython', 'notebook', '--script', '--notebook-dir', path, '--pylab', 'inline'])
    try:
        c.wait()
    except KeyboardInterrupt:
        c.terminate()


def dict_to_str(dic):
        """join dictionary study_data into a string with that study_data"""
        return "_".join([k + ":" + str(v) for (k, v) in dic.items()])


#def path_to_this_file():
#
#    return os.path.join(os.path.dirname(__file__))