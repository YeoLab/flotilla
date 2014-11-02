"""
special tricks for opening flotilla with docker in OS X

for linux, this needs to use sudo
"""

import time
import subprocess
import os
import sys
import signal
import argparse
import json

DEFAULT_FLOTILLA_VERSION = "dev"
DEFAULT_FLOTILLA_NOTEBOOK_DIR = "~/flotilla_notebooks"
DEFAULT_FLOTILLA_PROJECTS_DIR = "~/flotilla_projects"
DEFAULT_MEMORY_REQUIREMENT = 3500

class CommandLine(object):
    def __init__(self, inOpts=None):
        self.parser = parser = argparse.ArgumentParser(
            description='Start flotilla with docker.')

        parser.add_argument('--branch', required=False,
                            type=str, action='store',
                            default=DEFAULT_FLOTILLA_VERSION,
                            help="branch of flotilla to "
                                 "use from dockerhub, default:{}".format(DEFAULT_FLOTILLA_VERSION))

        parser.add_argument('--notebook_dir', required=False,
                            type=str, action='store',
                            default=DEFAULT_FLOTILLA_NOTEBOOK_DIR,
                            help="local directory to place/read notebooks:{}".format(DEFAULT_FLOTILLA_NOTEBOOK_DIR))

        parser.add_argument('--flotilla_packages', required=False,
                            type=str, action='store',
                            default=DEFAULT_FLOTILLA_PROJECTS_DIR,
                            help="local directory to place/read "
                            "flotilla packages:{}".format(DEFAULT_FLOTILLA_PROJECTS_DIR))

        parser.add_argument('--memory_request', required=False,
                            type=int, action='store',
                            default=DEFAULT_MEMORY_REQUIREMENT,
                            help="memory request for docker VM:{}".format(DEFAULT_MEMORY_REQUIREMENT))

        if inOpts is None:
            self.args = vars(self.parser.parse_args())
        else:
            self.args = vars(self.parser.parse_args(inOpts))

    def do_usage_and_die(self, str):
        '''
        If a critical error is encountered, where it is suspected that the
        program is not being called with consistent parameters or data, this
        method will write out an error string (str), then terminate execution
        of the program.
        '''
        import sys

        print >> sys.stderr, str
        self.parser.print_usage()
        return 2


# Class: Usage
class Usage(Exception):
    '''
    Used to signal a Usage error, evoking a usage statement and eventual
    exit when raised
    '''

    def __init__(self, msg):
        self.msg = msg



def waiter(signum, frame):
    pass

def is_docker_running():
    """
    open a subprocess and check that `boot2docker status` returns 'running'
    raises OSError if boot2docker command fails
    """
    with open("/dev/null", 'w') as junk:
        try:
            p = subprocess.Popen("boot2docker status", shell=True, stdout=subprocess.PIPE, stderr=junk)
        except OSError:
            print "There is a problem with the boot2docker installation"
            raise
        stdout, stderr = p.communicate()
        if stdout == "running\n":
            return True
        else:
            return False


class Boot2DockerRunner(object):

    def assert_minimum_memory_allocated(self, minimum=DEFAULT_MEMORY_REQUIREMENT):
        """
        check that the boot2docker app has started with sufficient memory
        miniumim = minimum memory required, in MB
        """


        p = subprocess.Popen("boot2docker info", shell=True, stdout=subprocess.PIPE)
        boot2docker_state = json.loads("".join(p.stdout.readlines()))
        if boot2docker_state['Memory'] < minimum:
            sys.stderr.write("WARNING: your boot2docker VM has been initialized with insufficient memory.\n")
        return boot2docker_state['Memory'] >= minimum

    def remove_boot2docker_image(self):
        sys.stderr.write("removing existing boot2docker image\n")
        p = subprocess.Popen("boot2docker delete", shell=True)
        p.wait()

    def initialize_boot2docker(self, memory=DEFAULT_MEMORY_REQUIREMENT):
        """set docker VM with memory allocation"""

        sys.stderr.write("initializing boot2docker VM with {}MB memory (yah, it needs a lot)\n\"\
        \"please wait...\n".format(memory))
        p = subprocess.Popen("boot2docker init -m {}".format(memory), shell=True)
        p.wait()

    def reinitialize_boot2docker(self, memory=DEFAULT_MEMORY_REQUIREMENT):
        """reset docker VM with memory allocation"""
        self.remove_boot2docker_image()
        self.initialize_boot2docker(memory)

    def __init__(self, memory=DEFAULT_MEMORY_REQUIREMENT, keep_docker_running=False):
        #keep docker open after exit
        self.keep_docker_running = keep_docker_running
        self.memory_req = memory

    def __enter__(self):
        try:
            subprocess.check_call("boot2docker status", shell=True)
        except:
            self.initialize_boot2docker(self.memory_req)

        if is_docker_running():
            # if docker was already running, don't stop it on exit
            self.keep_docker_running = True

        if not self.assert_minimum_memory_allocated(self.memory_req):
            self.reinitialize_boot2docker(self.memory_req)

        boot2docker_cmd = "boot2docker -m {} up".format(self.memory_req)
        sys.stderr.write("starting boot2docker with: {}".format(boot2docker_cmd))
        p = subprocess.Popen(boot2docker_cmd, shell=True, stdout=subprocess.PIPE)
        output = p.stdout.readlines()
        for line in output:
            line = line.strip()
            print line
            if "export" in line:
                export_this = line.split("export ")[1]
                variable, value = export_this.split("=")
                os.environ[variable] = value
        os.environ['DOCKER_IP'] = os.environ['DOCKER_HOST'].lstrip("tcp://").split(":")[0]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.keep_docker_running:
            p = subprocess.Popen("boot2docker stop", shell=True)
            try:
                sys.stderr.write("Shutting down boot2docker. Be slightly patient.")
                p.wait()
            except KeyboardInterrupt:
                signal.signal(signal.SIGTERM, waiter)


class FlotillaRunner(object):
    """Start docker flotilla, open the browser"""
    def __init__(self, flotilla_version=DEFAULT_FLOTILLA_VERSION,
                 notebook_dir=DEFAULT_FLOTILLA_NOTEBOOK_DIR,
                 flotilla_packages_dir=DEFAULT_FLOTILLA_PROJECTS_DIR,
                 memory_request=DEFAULT_MEMORY_REQUIREMENT):
        notebook_dir =os.path.abspath(os.path.expanduser(notebook_dir))
        flotilla_packages_dir = os.path.abspath(os.path.expanduser(flotilla_packages_dir))
        self.flotilla_version = flotilla_version
        self.flotilla_process = None
        self.flotilla_packages_dir = flotilla_packages_dir
        self.notebook_dir = notebook_dir
        self.memory_request = memory_request

        subprocess.call("docker pull mlovci/flotilla:%s" % flotilla_version, shell=True)

    def __enter__(self):
        docker_runner = "docker run -m \"{3}m\" -v {0}:/root/flotilla_projects " \
                               "-v {1}:/root/ipython " \
                               "-d -P -p 8888 " \
                               "mlovci/flotilla:{2}".format(self.flotilla_packages_dir,
                                                            self.notebook_dir,
                                                            self.flotilla_version,
                                                            self.memory_request)
        sys.stderr.write("running: {}\n".format(docker_runner))
        self.flotilla_process = subprocess.Popen(docker_runner,
                                                 shell=True, stdout=subprocess.PIPE)
        self.flotilla_container = self.flotilla_process.stdout.readlines()[0].strip()
        docker_port = subprocess.Popen("docker port {}".format(self.flotilla_container),
                                       shell=True,
                                       stdout=subprocess.PIPE)
        self.flotilla_port = docker_port.stdout.readlines()[0].split(":")[-1].strip()
        flotilla_url = 'http://{}:{}'.format(os.environ['DOCKER_IP'],  self.flotilla_port)
        sys.stderr.write("flotilla is running at: {}".format(flotilla_url))
        subprocess.call('open {}'.format(flotilla_url), shell=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        p = subprocess.Popen("docker stop {0} && docker rm {0}".format(self.flotilla_container), shell=True)
        try:
            sys.stderr.write("Shutting down notebook. Be slightly patient.")
            p.wait()
        except KeyboardInterrupt:
                signal.signal(signal.SIGTERM, waiter)


def main(flotilla_branch, flotilla_notebooks, flotilla_projects, memory_request):
    with Boot2DockerRunner(memory_request) as bd:
        with FlotillaRunner(flotilla_branch, flotilla_notebooks, flotilla_projects, memory_request) as fr:
            print "Use Ctrl-C once, and only once, to exit"
            while True:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    exit(0)



if __name__ == '__main__':
    try:
        cl = CommandLine()
        main(cl.args['branch'], cl.args['notebook_dir'],
             cl.args['flotilla_packages'], cl.args['memory_request'])

    except Usage, err:
        cl.do_usage_and_die()