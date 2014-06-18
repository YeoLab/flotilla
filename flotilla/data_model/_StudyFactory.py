__author__ = 'olga'

import sys
import os
from  ..util import install_development_package
import pandas as pd
import subprocess

class StudyFactory(object):

    #'min_samples','species',
                       #'sample_metadata_filename', 'event_metadata_filename', 'expression_data_filename',
                       #'splicing_data_filename', 'expression_data_filename', 'gene_metadata_filename',
                       #'event_metadata_filename',
                       # 'default_group_id', 'default_group_ids', 'default_list_id', 'default_list_ids']

    def __init__(self):
        self.minimal_study_parameters = set()
        self.new_study_params = set()
        self.getters = []

    def write_package(self, study_name, write_location=None, install=False):
        write_these = self.minimal_study_parameters

        data_resources = ['sample_metadata', 'expression_df', 'splicing_df', 'event_metadata']

        [self.minimal_study_parameters.add(i) for i in write_these]
        self.validate_params()

        new_package_data_location = self._clone_barebones(study_name, write_location=write_location)

        #new_package_data_location is os.path.join(write_location, study_name)

        self._write_params_file(new_package_data_location, params_to_write = write_these)
        for resource_name in data_resources:
            data = getattr(self, resource_name)
            try:
                self._add_package_data_resource(resource_name, data, new_package_data_location,
                file_write_mode='tsv')
            except:
                sys.stderr.write("couldn't add data resource: %s\n" % resource_name)

        if install:
            install_development_package(os.path.abspath(write_location))

    def _clone_barebones(self, study_name, write_location=None):
        import flotilla
        flotilla_install_location = os.path.dirname(os.path.abspath(flotilla.__file__))
        test_package_location = os.path.join(flotilla_install_location, "cargo/cargo_data/"\
                                                                        "barebones_project")
        starting_position = os.getcwd()
        try:
            if write_location is None:
                write_location = os.path.abspath(starting_position)
            else:
                #TODO: check whether user specificed a real writable location
                pass
            if os.path.exists(write_location):
                raise Exception("do not use an existing path for write_location")

            subprocess.call(['git clone -b barebones %s %s' % (test_package_location, write_location)], shell=True)
            os.chdir(write_location)
            subprocess.call(['git mv barebones_project %s' % study_name], shell=True)
            with open("setup.py", 'r') as f:
                setup_script = f.readlines()


            with open("setup.py", 'w') as f:
                for i in setup_script:

                    f.write(i.replace("barebones_project", study_name))
            os.chdir(starting_position)

        except:
            sys.stderr.write("error, did not complete cloning")
            os.chdir(starting_position)
            raise
        return os.path.join(write_location, study_name)

    def _write_params_file(self, package_location, params_to_write):

        import os
        with open(os.path.join(package_location, "params.py"), 'w') as f:
            f.write("from .study_data import study_data_dir\n\n")
            f.write("")
            for param in params_to_write:

                try:
                    f.write("#%s" % self.doc(param))
                except:
                    pass
                value = getattr(self, param)
                if "filename" in param:
                    if value is not None:
                        value = self._to_base_file_tuple(value)

                f.write("%s = %s\n\n" % (param, repr(value)))

    def _to_base_file_tuple(self, tup):
        """for making new packages, auto-loadable data!"""
        assert len(tup) == 2
        return "os.path.join(study_data_dir, %s)" % os.path.basename(tup[0]), tup[1]

    def _add_package_data_resource(self, file_name, data_df, toplevel_package_dir, file_write_mode="tsv"):
        writer = getattr(self, "_write_" + file_write_mode)
        file_base = os.path.basename(file_name)
        rsc_file = os.path.join(toplevel_package_dir, "study_data", file_base + "." + file_write_mode)
        writer(data_df, rsc_file)
        return (rsc_file, file_write_mode)

    def _set(self, k, v):
        """set attributes, warn if re-setting"""

        try:
            assert not hasattr(self, k)
        except:
            write_me = "WARNING: over-writing parameter " + k + "\n" #+ \
                       #str(self.__getattribute__(k)) + \
                       #"\n new:" + str(v)
            sys.stderr.write(write_me)
        super(StudyFactory, self).__setattr__(k,v)

    def update(self, dict):
        [self._set(k,v) for (k,v) in dict.items() if not k.startswith("_")] #skip private variables

    def validate_params(self):
        """make sure that all necessary attributes are present"""
        for param in self.minimal_study_parameters:
            try:
                assert hasattr(self, param)
            except:
                raise AssertionError("missing minimal parameter %s" % param)

    _accepted_filetypes = []
    _accepted_filetypes.append('pickle_df')
    def _load_pickle_df(self, file_name):
        return pd.read_pickle(file_name)

    def _write_pickle_df(self, df, file_name):
        df.to_pickle(file_name)


    _accepted_filetypes.append('gzip_pickle_df')
    def _load_gzip_pickle_df(self, file_name):
        import gzip, cPickle
        with gzip.open(file_name, 'r') as f:
            return cPickle.load(f)

    def _write_gzip_pickle_df(self, df, file_name):
        import tempfile
        tmpfile_h, tmpfile = tempfile.mkstemp()
        df.to_pickle(tmpfile)
        import subprocess
        subprocess.call(['gzip -f %s' % tempfile])
        subprocess.call(['mv %s %s' % (tempfile, file_name)])


    _accepted_filetypes.append('tsv')
    def _load_tsv(self, file_name):
        return pd.read_table(file_name, index_col=0)

    def _write_tsv(self, df, file_name):
        df.to_csv(file_name, sep='\t')


    _accepted_filetypes.append('csv')
    def _load_csv(self, file_name):
        return pd.read_csv(file_name, index_col=0)

    def _write_csv(self, df, file_name):
        df.to_csv(file_name)


    def _get_loading_method(self, file_name):
        """loading_methods for loading from file"""
        return getattr(self, "_load_" + file_name)

    def load(self, file_name, file_type='pickle_df'):
        return self._get_loading_method(file_type)(file_name)

    def register_new_getter(self, getter_name, **kwargs):
        self.getters.append((kwargs.copy(), getter_name))

    def apply_getters(self):
        """
        update instance namespace with outputs of registered getters.
        """
        for data, getter in self.getters:
            #formerly explicitly set things
            for (k,v) in getter(**data).items():
                self._set(k,v)
        self.getters = [] #reset, getters only need to run once.

    def _example_getter(self, named_attribute=None):
        #perform operations on named inputs
        #return named outputs

        return {'output1':None}