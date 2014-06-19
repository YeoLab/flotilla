"""
Data models for "studies" studies include attributes about the data and are
heavier in terms of data load
"""

from collections import defaultdict
import os
import pandas as pd
import subprocess
import sys
import warnings

from .expression import ExpressionData
from .splicing import SplicingData
from .experiment_design import ExperimentDesignData
from ..util import install_development_package
from ..visualize import NetworkerViz, PredictorViz
from ..visualize.color import red, blue, green
from ..visualize.ipython_interact import Interactive

# import flotilla
# FLOTILLA_DIR = os.path.dirname(flotilla.__file__)

class StudyFactory(object):

    #'min_samples','species',
                       #'sample_metadata_filename', 'event_metadata_filename', 'expression_data_filename',
                       #'splicing_data_filename', 'expression_data_filename', 'gene_metadata_filename',
                       #'event_metadata_filename',
                       # 'default_group_id', 'default_group_ids', 'default_list_id', 'default_list_ids']
    _accepted_filetypes = 'tsv'
    # _accepted_filetypes = ['pickle_df', 'gzip_pickle_df', 'tsv', 'csv']

    def __init__(self):
        self.minimal_study_parameters = set()
        self.new_study_params = set()
        self.getters = []

    def __setattr__(self, key, value):
        """Check if the attribute already exists and warns on overwrite.
        """
        if hasattr(self, key):
            warnings.warn('Over-writing attribute {}'.format(key))
        super(StudyFactory, self).__setattr__(key, value)

    def write_package(self, study_name, where=None, install=False):
        write_these = self.minimal_study_parameters

        data_resources = ['phenotype_data', 'expression_df', 'splicing_df', 'event_metadata']

        self.minimal_study_parameters.update(write_these)
        self.validate_params()

        new_package_data_location = self._clone_barebones(study_name, write_location=where)

        #new_package_data_location is os.path.join(where, study_name)

        self._write_params_file(new_package_data_location, params_to_write = write_these)
        for resource_name in data_resources:
            data = getattr(self, resource_name)
            try:
                self._add_package_data_resource(resource_name, data, new_package_data_location,
                file_write_mode='tsv')
            except:
                sys.stderr.write("couldn't add data resource: %s\n" % resource_name)

        if install:
            install_development_package(os.path.abspath(where))

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
                #TODO.md: check whether user specificed a real writable location
                pass
            try:
                path_exists = os.path.exists(write_location)
            except TypeError:
                # Need this for testing, which creates a LocalPath instead of
                #  a string
                path_exists = False
                write_location = str(write_location)

            if path_exists:
                raise Exception("do not use an existing path for write_location")

            subprocess.call(['git clone -b barebones %s %s' % (test_package_location, write_location)], shell=True)
            os.chdir(write_location)
            subprocess.call(['git mv barebones_project %s' % study_name], shell=True)
            with open("{}/setup.py".format(FLOTILLA_DIR), 'r') as f:
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

    def _add_package_data_resource(self, file_name, data_df,
                                   toplevel_package_dir,
                                   file_write_mode="tsv"):
        writer = getattr(self, "_write_" + file_write_mode)
        file_base = os.path.basename(file_name)
        rsc_file = os.path.join(toplevel_package_dir, "study_data", file_base + "." + file_write_mode)
        writer(data_df, rsc_file)
        return (rsc_file, file_write_mode)

    # def _set(self, k, v):
    #     """set attributes, warn if re-setting"""
    #
    #     try:
    #         assert not hasattr(self, k)
    #     except:
    #         write_me = "WARNING: over-writing parameter " + k + "\n" #+ \
    #                    #str(self.__getattribute__(k)) + \
    #                    #"\n new:" + str(v)
    #         sys.stderr.write(write_me)
    #     super(StudyFactory, self).__setattr__(k, v)

    # def update(self, dict):
    #     [self._set(k,v) for (k,v) in dict.items() if not k.startswith("_")] #skip private variables

    def validate_params(self):
        """make sure that all necessary attributes are present"""
        for param in self.minimal_study_parameters:
            try:
                x = getattr(self, param)
            except KeyError:
                raise AssertionError("Missing minimal parameter %s" % param)

    def _load_pickle_df(self, file_name):
        return pd.read_pickle(file_name)

    def _write_pickle_df(self, df, file_name):
        df.to_pickle(file_name)

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

    def _load_tsv(self, file_name):
        return pd.read_table(file_name, index_col=0)

    def _write_tsv(self, df, file_name):
        df.to_csv(file_name, sep='\t')

    def _load_csv(self, file_name):
        return pd.read_csv(file_name, index_col=0)

    def _write_csv(self, df, file_name):
        df.to_csv(file_name)

    def _get_loading_method(self, file_name):
        """loading_methods for loading from file"""
        return getattr(self, "_load_" + file_name)

    def load(self, file_name, file_type='pickle_df'):
        return self._get_loading_method(file_type)(file_name)

    # def register_new_getter(self, getter_name, **kwargs):
    #     self.getters.append((kwargs.copy(), getter_name))
    #
    # def apply_getters(self):
    #     """
    #     update instance namespace with outputs of registered getters.
    #     """
    #     for data, getter in self.getters:
    #         #formerly explicitly set things
    #         for (k,v) in getter(**data).items():
    #             self._set(k,v)
    #     self.getters = [] #reset, getters only need to run once.
    #
    # def _example_getter(self, named_attribute=None):
    #     """Perform operations on named inputs and return named outputs
    #     """
    #     return {'output1': None}


class Study(StudyFactory):
    """
    store essential data associated with a study. Users specify how to build the
    necessary components from project-specific getters (see barebones_project
    for example getters)
    """
    default_feature_set_ids = []
    def __init__(self, experiment_design_data, expression_data=None, splicing_data=None,
                 expression_feature_data=None, splicing_feature_data=None,
                 load_cargo=False, drop_outliers=False,
                 default_group_id=None, default_group_ids=None,
                 default_list_id='variant', default_list=('variant'),
                 default_genes=('variant'),
                 default_events=('variant'), species=None):
        """Construct a biological study

        This class only accepts data, no filenames. All data must already
        have been loaded in.

        Parameters
        ----------
        experiment_design_data : pandas.DataFrame
            Only required parameter. Samples as the index, with
        expression_data : pandas.DataFrame
            Samples x feature dataframe of gene expression measurements,
            e.g. from an RNA-Seq or a microarray experiment. Assumed to be
            log-normal (i.e. not log-transformed)
        expression_feature_data : pandas.DatFrame
            features x other_features dataframe describing other parameters
            of the gene expression features, e.g. mapping Ensembl IDs to gene
            symbols or gene biotypes.

        Returns
        -------


        Raises
        ------

        """
        super(Study, self).__init__()
        # if params_dict is None:
        #     params_dict = {}
        # self.update(params_dict)
        # self.initialize_required_getters()
        # self.apply_getters()
        self.species = species
        
        self._initialize_all_data(experiment_design_data,
                                   expression_data,
                                   splicing_data, expression_feature_data,
                                   splicing_feature_data,
                                   load_cargo=load_cargo,
                                  drop_outliers=drop_outliers)
        sys.stderr.write("subclasses initialized\n")
        self.validate_params()
        sys.stderr.write("package validated\n")

    def __add__(self, other):
        """Sanely concatenate one or more Study objects
        """
        raise NotImplementedError
        self.phenotype_data = pd.concat([self.phenotype_data,
                                          other.phenotype_data])
        self.expression.data = pd.concat([self.expression.data,
                                          other.phenotype_data])
        # self.species = # dict of sample ids to species?

    # def initialize_required_getters(self):
    #     if self.sample_metadata_filename is not None and self.event_metadata_filename is not None:
    #
    #         self.register_new_getter(self.get_metadata,
    #                                      sample_metadata_filename=self.sample_metadata_filename,
    #                                      gene_metadata_filename=self.gene_metadata_filename,
    #                                      event_metadata_filename=self.event_metadata_filename,
    #                                      )
    #     else:
    #         raise RuntimeError("at least s-et sample_metadata_filename and event_metadata_filename")
    #
    #     if self.expression_data_filename is not None:
    #         self.register_new_getter(self.get_expression_data, expression_data_filename=self.expression_data_filename)
    #
    #     else:
    #         raise RuntimeError("at least set expression_data_filename")
    #
    #     if self.splicing_data_filename is not None:
    #         self.register_new_getter(self.get_splicing_data, splicing_data_filename=self.splicing_data_filename)
    #     else:
    #         raise RuntimeError("at least set splicing_data_filename")

    def _initialize_all_data(self, phenotype_data,
                             expression_data, splicing_data,
                             expression_feature_data,
                             splicing_feature_data, load_cargo=False,
                             drop_outliers=False):
        """Initialize all the datasets
        """
        #TODO.md: would be great if this worked, untested:
        #for subclass in self.subclasses:
        #    initializer = getattr(self, 'intialize_', subclass, '_subclass')
        #    initializer(self)


        self._initialize_phenotype_data(phenotype_data)
        sys.stderr.write("initializing expression\n")
        self._initialize_expression(expression_data,
                                    expression_feature_data,
                                    load_cargo=load_cargo,
                                    drop_outliers=drop_outliers)
        try:
            sys.stderr.write("initializing splicing\n")
            self._initialize_splicing(splicing_data, splicing_feature_data,
                                      load_cargo=False,
                                      drop_outliers=drop_outliers)
        except:
            warnings.warn("Failed to load splicing")

    def _initialize_phenotype_data(self, phenotype_data):
        #TODO.md: this should be an actual data_model.*Data type, but now it's just set by a loader
        sys.stderr.write("initializing phenotype data\n")
        self.phenotype = ExperimentDesignData(phenotype_data)

        self._set_plot_colors()
        self._set_plot_markers()

    def _set_plot_colors(self):
        """If there is a column 'color' in the sample metadata, specify this
        as the plotting color
        """
        try:
            self._default_reducer_kwargs.update(
                {'colors_dict': self.phenotype_data.color})
            self._default_plot_kwargs.update(
                {'color': self.phenotype_data.color.tolist()})
        except AttributeError:
            sys.stderr.write("There is no column named 'color' in the "
                             "metadata, defaulting to blue for all samples\n")
            self._default_reducer_kwargs.update(
                {'colors_dict': defaultdict(lambda: blue)})

    def _set_plot_markers(self):
        """If there is a column 'marker' in the sample metadata, specify this
        as the plotting marker (aka the plotting shape). Only valid matplotlib
        symbols are allowed. See http://matplotlib.org/api/markers_api.html
        for a more complete description.
        """
        try:
            self._default_reducer_kwargs.update(
                {'markers_dict': self.phenotype_data.marker})
            self._default_plot_kwargs.update(
                {'marker': self.phenotype_data.marker.tolist()})
        except AttributeError:
            sys.stderr.write("There is no column named 'marker' in the sample "
                             "metadata, defaulting to a circle for all "
                             "samples\n")
            self._default_reducer_kwargs.update({'markers_dict':
                                                     defaultdict(lambda: 'o')})

    def _initialize_expression(self, expression_data,
                               expression_feature_data,
                               load_cargo=True, drop_outliers=True,
                               force=False):
        """Initialize ExpressionData object

        Parameters
        ----------
        expression_data : pandas.DataFrame
            Samples x feature dataframe of gene expression measurements,
            e.g. from an RNA-Seq or a microarray experiment. Assumed to be
            log-normal (i.e. not log-transformed)
        expression_feature_data : pandas.DatFrame
            features x other_features dataframe describing other parameters
            of the gene expression features, e.g. mapping Ensembl IDs to gene
            symbols or gene biotypes.
        load_cargo : bool
            Whether or not to load the "cargo" (aka feature metadata)
            associated with gene expression in this species
        drop_outliers : bool
            Whether or not to remove the samples specified as outliers in the
            phenotype data
        force : bool
            Whether or not to overwrite an existing 'expression' object (if
            it exists)

        Raises
        ------
        AttributeError if "expression" attribute already exists
        """
        # [self.minimal_study_parameters.add(i) for i in ['expression_df', 'expression_metadata']]
        # self.validate_params()

        #TODO.md:don't over-write self.expression
        if hasattr(self, 'expression') and not force:
            raise AttributeError('Already have attribute "expression", '
                                 'not overwriting. Set "force=True" to force '
                                 'overwriting')
        else:
            self.expression = ExpressionData(data=expression_data,
                                             feature_data=expression_feature_data,
                                             load_cargo=load_cargo,
                                             drop_outliers=drop_outliers,
                                             species=self.species)
            self.expression.networks = NetworkerViz(self.expression)
            self.default_feature_set_ids.extend(self.expression.feature_sets
                                            .keys())

    def _initialize_splicing(self, splicing_data, splicing_feature_data,
                             load_cargo=False, drop_outliers=True,
                             force=False):
        """Initialize SplicingData object

        Parameters
        ----------
        splicing_data : pandas.DataFrame
            Samples x features dataframe of "percent spliced-in" PSI scores
        splicing_feature_data : pandas.DataFrame
            Samples x features dataframe describing features of the splicing
            events
        load_cargo : bool
            Whether or not to load the "cargo" (aka feature metadata)
            associated with alternative splicing in this species
        drop_outliers : bool
            Whether or not to remove the samples specified as outliers in the
            phenotype data
        force : bool
            Whether or not to overwrite an existing 'expression' object (if
            it exists)

        Raises
        ------
        AttributeError if "splicing" attribute already exists
        """

        # [self.minimal_study_parameters.add(i) for i in ['splicing_df', 'event_metadata']]
        # self.validate_params()
        #TODO.md:don't over-write self.splicing
        if hasattr(self, 'splicing') and not force:
            raise AttributeError('Already have attribute "splicing" set, '
                                 'not overwriting. Set "force=True" to force '
                                 'overwriting.')
        else:
            self.splicing = SplicingData(data=splicing_data,
                                         feature_data=splicing_feature_data,
                                         load_cargo=load_cargo,
                                         drop_outliers=drop_outliers,
                                         species=self.species)
            self.splicing.networks = NetworkerViz(self.splicing)


    # def get_expression_data(self, expression_data_filename):
    #     return {'expression_df': self.load(*expression_data_filename)}
    #
    #
    # def get_splicing_data(self, splicing_data_filename):
    #     return {'splicing_df': self.load(*splicing_data_filename)}
    #
    # def get_metadata(self, sample_metadata_filename=None,
    #                  gene_metadata_filename=None,
    #                  event_metadata_filename=None):
    #
    #     metadata = {'sample':None,
    #                'gene':None,
    #                'event':None}
    #     try:
    #         if sample_metadata_filename is not None:
    #             metadata['sample'] = self.load(*sample_metadata_filename)
    #         if event_metadata_filename is not None:
    #             metadata['event'] = self.load(*event_metadata_filename)
    #         if gene_metadata_filename is not None:
    #             metadata['gene'] = self.load(*gene_metadata_filename)
    #
    #     except Exception as E:
    #         sys.stderr.write("error loading descriptors: %s, \n\n .... entering pdb ... \n\n" % E)
    #         raise E
    #
    #     return {'phenotype_data': metadata['sample'],
    #             'feature_data': metadata['gene'],
    #             'event_metadata': metadata['event'],
    #             'expression_metadata': None}

    def main(self):
        raise NotImplementedError
        #TODO.md: make this an entry-point, parse flotilla package to load from cmd line, do something
        #this is for the user... who will know little to nothing about queues and the way jobs are done on the backend

        usage = "run_flotilla_cmd cmd_name runner_name"
        # def runner_concept(self, flotilla_package_target = "barebones_package", tool_name):
        #
        #     #a constructor for a new study, takes a long time and maybe runs in parallel. Probably on a cluster...
        #     #same as `import flotilla_package_target as imported_package`
        #
        #     imported_package = __import__(flotilla_package_target)
        #     study = Study.__new__()
        #
        #     #should use importlib though...
        #
        #
        #     if this_is_not a parallel process
        #         try:
        #             make a new runner_name.lock file
        #         except: #exists(runner_name.lock)
        #             raise RuntimeError
        #
        #     study.do_something()


    def detect_outliers(self):
        """Detects outlier cells from expression, mapping, and splicing
        study_data and labels the outliers as such for future analysis.

        Parameters
        ----------
        self

        Returns
        -------


        Raises
        ------

        """
        #TODO.md: Boyko/Patrick please implement
        raise NotImplementedError


    def jsd(self):
        """Performs Jensen-Shannon Divergence on both splicing and expression study_data

        Jensen-Shannon divergence is a method of quantifying the amount of
        change in distribution of one measurement (e.g. a splicing event or a
        gene expression) from one celltype to another.
        """
        raise NotImplementedError
        #TODO.md: Check if JSD has not already been calculated (cacheing or memoizing)

        self.expression.jsd()
        self.splicing.jsd()


    def normalize_to_spikein(self):
        raise NotImplementedError

    def compute_expression_splicing_covariance(self):
        raise NotImplementedError

    def jsd(self):
        raise NotImplementedError

    def plot_pca(self, data_type='expression', x_pc=1, y_pc=2, **kwargs):
        """Performs PCA on both expression and splicing study_data
        """
        if data_type == "expression":
            self.expression.plot_dimensionality_reduction(x_pc=x_pc, y_pc=y_pc,
                                                          **kwargs)
        elif data_type == "splicing":
            self.splicing.plot_dimensionality_reduction(x_pc=x_pc, y_pc=y_pc,
                                                        **kwargs)

    def plot_graph(self, data_type='expression', **kwargs):
        if data_type == "expression":
            self.expression.networks.draw_graph(**kwargs)

        elif data_type == "splicing":
            self.splicing.networks.draw_graph(**kwargs)

    def plot_classifier(self, data_type='expression', **kwargs):
        """
        """
        if data_type == "expression":
            self.expression.plot_classifier(**kwargs)
        elif data_type == "splicing":
            self.splicing.plot_classifier(**kwargs)

    def plot_regressor(self, data_type='expression', **kwargs):
        """
        """
        raise NotImplementedError
        if data_type == "expression":
            self.expression.plot_regressor(**kwargs)
        elif data_type == "splicing":
            self.splicing.plot_regressor(**kwargs)

# Add interactive visualizations
Study.interactive_classifier = Interactive.interactive_classifier
Study.interactive_graph = Interactive.interactive_graph
Study.interactive_pca = Interactive.interactive_pca
Study.interactive_localZ = Interactive.interactive_localZ
