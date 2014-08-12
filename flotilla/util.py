"""

general utilities

"""

from functools import wraps
import errno
import os
import re
import signal
import sys
import subprocess
import functools
import time


class TimeoutError(Exception):
    """
    From:
    http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    """
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


def serve_ipython():
    try:

        assert len(sys.argv) == 2
        path = sys.argv[1]
        assert os.path.exists(sys.argv[1])

    except:
        raise ValueError("specify a notebook directory as the first and only "
                         "argument")

    c = subprocess.Popen(['ipython', 'notebook', '--script', '--notebook-dir',
                          path])
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


def memoize(obj):
    """
    'memoize' aka remember the output from a function and return that,
    rather than recalculating

    Stolen from:
    https://wiki.python.org/moin/PythonDecoratorLibrary#CA-237e205c0d5bd1459c3663a3feb7f78236085e0a_1

    do_not_memoize : bool
        IF this is a keyword argument (kwarg) in the function, and it is true,
        then just evaluate the function and don't memoize it.
    """
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if 'do_not_memoize' in kwargs and kwargs['do_not_memoize']:
            return obj(*args, **kwargs)
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer


class cached_property(object):
    '''Decorator for read-only properties evaluated only once within TTL period.

    It can be used to created a cached property like this::

        import random

        # the class containing the property must be a new-style class
        class MyClass(object):
            # create property whose value is cached for ten minutes
            @cached_property(ttl=600)
            def randint(self):
                # will only be evaluated every 10 min. at maximum.
                return random.randint(0, 100)

    The value is cached  in the '_cache' attribute of the object instance that
    has the property getter method wrapped by this decorator. The '_cache'
    attribute value is a dictionary which has a key for every property of the
    object which is wrapped by this decorator. Each entry in the cache is
    created only when the property is accessed for the first time and is a
    two-element tuple with the last computed property value and the last time
    it was updated in seconds since the epoch.

    The default time-to-live (TTL) is 300 seconds (5 minutes). Set the TTL to
    zero for the cached value to never expire.

    To expire a cached property value manually just do::

        del instance._cache[<property name>]

    Stolen from:
    https://wiki.python.org/moin/PythonDecoratorLibrary#Cached_Properties

    '''

    def __init__(self, ttl=0):
        self.ttl = ttl

    def __call__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__
        return self

    def __get__(self, inst, owner):
        now = time.time()
        try:
            value, last_update = inst._cache[self.__name__]
            if self.ttl > 0 and now - last_update > self.ttl:
                raise AttributeError
        except (KeyError, AttributeError):
            value = self.fget(inst)
            try:
                cache = inst._cache
            except AttributeError:
                cache = inst._cache = {}
            cache[self.__name__] = (value, now)
        return value


def as_numpy(x):
    """Given either a pandas dataframe or a numpy array, always return a
    numpy array.
    """
    try:
        # Pandas DataFrame
        return x.values
    except AttributeError:
        # Numpy array
        return x


def natural_sort(l):
    """
    From
    http://stackoverflow.com/a/4836734
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)