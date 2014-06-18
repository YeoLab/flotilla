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
import functools

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


def serve_ipython():
    try:

        assert len(sys.argv) == 2
        path = sys.argv[1]
        assert os.path.exists(sys.argv[1])

    except:
        raise ValueError("specify a notebook directory as the first and only argument")

    c = subprocess.Popen(['ipython', 'notebook', '--script', '--notebook-dir', path])
    try:
        c.wait()
    except KeyboardInterrupt:
        c.terminate()


def dict_to_str(dic):
        """join dictionary study_data into a string with that study_data"""
        return "_".join([k + ":" + str(v) for (k, v) in dic.items()])

def install_development_package(package_location):
    original_location = os.getcwd()
    os.chdir(package_location)
    subprocess.call(['pip install -e %s' % package_location], shell=True)
    os.chdir(original_location)

#def path_to_this_file():
#
#    return os.path.join(os.path.dirname(__file__))

def memoize(obj):
    """
    'memoize' aka remember the output from a function and return that,
    rather than recalculating

    Stolen from:
    https://wiki.python.org/moin/PythonDecoratorLibrary#CA-237e205c0d5bd1459c3663a3feb7f78236085e0a_1
    """
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer