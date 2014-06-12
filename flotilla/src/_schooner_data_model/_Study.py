"""
Data models for "studies" studies include attributes about the data and are
heavier in terms of data load
"""

from .._submaraine_viz import NetworkerViz, PredictorViz, plt
import sys, os, subprocess
from _ExpressionData import ExpressionData
from _SplicingData import SplicingData
import pandas as pd


class BaseStudy(object):

    def __init__(self):
        self.initialize_base()

    def initialize_base(self):
        self.minimal_study_parameters = set(['study_data_dir'])

    def _set(self, k, v):
        """set attributes, warn if re-setting"""

        try:
            assert not hasattr(self, k)
        except:
            write_me = "WARNING: over-writing parameter " + k + "\n" #+ \
                       #str(self.__getattribute__(k)) + \
                       #"\n new:" + str(v)
            sys.stderr.write(write_me)
        super(BaseStudy, self).__setattr__(k,v)

    def update(self, dict):
        [self._set(k,v) for (k,v) in dict.items() if not k.startswith("_")] #skip private variables

    def validate_params(self):
        """make sure that all necessary attributes are present"""
        for param in self.minimal_study_parameters:
            try:
                assert hasattr(self, param)
            except:
                raise AssertionError("missing minimal parameter %s" % param)

class DataOnDiskManager(BaseStudy):

    """
    manage constructing data from files on disk
    getters accept named kwargs, they return named outputs,
    the outputs are then added as named instance attributes
    """

    def initialize_datamanager(self):
        self.getters=[]

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
        self.initialize_datamanager() #reset, getters only need to run once.

    def _example_getter(self, named_attribute=None):
        #perform operations on named inputs
        #return named outputs

        return {'output1':None}

    def get_metadata(self, sample_metadata_filename=None, gene_metadata_filename=None,
                     event_metadata_filename=None):

        metadata = {'sample':None,
                   'gene':None,
                   'event':None}
        try:
            if sample_metadata_filename is not None:
                metadata['sample'] = self.load(*sample_metadata_filename)
            if event_metadata_filename is not None:
                metadata['event'] = self.load(*event_metadata_filename)
            if gene_metadata_filename is not None:
                metadata['gene'] = self.load(*gene_metadata_filename)

        except Exception as E:
            sys.stderr.write("error loading descriptors: %s, \n\n .... entering pdb ... \n\n" % E)
            raise E

        return {'sample_metadata': metadata['sample'],
                'gene_metadata': metadata['gene'],
                'event_metadata': metadata['event'],
                'expression_metadata': None}

    def get_splicing_data(self, splicing_data_filename):
        return {'splicing_df': self.load(*splicing_data_filename)}

    def get_expression_data(self, expression_data_filename):
        return {'expression_df': self.load(*expression_data_filename)}

    def get_transcriptome_data(self, expression_data_filename, splicing_data_filename):
        try:
            splicing = self.get_splicing_data(splicing_data_filename)['splicing_df']
            expression = self.get_expression_data(expression_data_filename)['expression_df']
            sparse_expression = expression[expression > 0]

        except Exception as E:
            sys.stderr.write("error loading transcriptome data: %s, \n\n .... entering pdb ... \n\n" % E)
            raise E

        return {'splicing_df': splicing,
                'expression_df': expression,
                'sparse_expression_df': sparse_expression}

    def doc(self):
        raise NotImplementedError
        self.docs = {
            'min_samples': 'minimum number of samples for analyses',
            'sample_metadata_filename': 'sample metadata',
            'event_metadata_filename': 'splicing feature metadata',
            'expression_data_filename': 'expression data filename',
            #TODO: fill this in
        }

class StudyData(DataOnDiskManager):
    """
    store essential data associated with a study. Users specify how to build the necessary components from
    project-specific getters (see barebones_project for example getters)
    """

    def __init__(self, params_dict=None, load_cargo=False, drop_outliers=False):
        self.initialize_base()
        if params_dict is None:
            params_dict = {}
        self.update(params_dict)
        self.initialize_datamanager()
        self.initialize_required_getters()
        self.apply_getters()
        self.initialize_all_subclasses(load_cargo=load_cargo, drop_outliers=drop_outliers)
        sys.stderr.write("subclasses initialized\n")
        self.validate_params()
        sys.stderr.write("package validated\n")

    def initialize_required_getters(self):

        if self.sample_metadata_filename is not None and self.event_metadata_filename is not None:

            self.register_new_getter(self.get_metadata,
                                         sample_metadata_filename=self.sample_metadata_filename,
                                         gene_metadata_filename=self.gene_metadata_filename,
                                         event_metadata_filename=self.event_metadata_filename,
                                         )
        else:
            raise RuntimeError("at least s-et sample_metadata_filename and event_metadata_filename")

        if self.expression_data_filename is not None:
            self.register_new_getter(self.get_expression_data, expression_data_filename=self.expression_data_filename)

        else:
            raise RuntimeError("at least set expression_data_filename")

        if self.splicing_data_filename is not None:
            self.register_new_getter(self.get_splicing_data, splicing_data_filename=self.splicing_data_filename)
        else:
            raise RuntimeError("at least set splicing_data_filename")

    def initialize_all_subclasses(self, load_cargo=False, drop_outliers=False):
        """
        run all initializers
        """

        #TODO: would be great if this worked, untested:
        #for subclass in self.subclasses:
        #    initializer = getattr(self, 'intialize_', subclass, '_subclass')
        #    initializer(self)

        sys.stderr.write("initializing metadata\n")
        self.initialize_sample_metadata_subclass(load_cargo=load_cargo, drop_outliers=drop_outliers)
        sys.stderr.write("initializing expression\n")
        self.initialize_expression_subclass(load_expression_cargo=load_cargo, drop_outliers=drop_outliers)
        try:
            sys.stderr.write("initializing splicing\n")
            self.initialize_splicing_subclass(load_splicing_cargo=False, drop_outliers=drop_outliers)
        except:
            sys.stderr.write("failed to load splicing")

    def initialize_sample_metadata_subclass(self, **kwargs):
        #TODO: this should be an actual schooner.*Data type, but now it's just set by a loader
        assert hasattr(self, 'sample_metadata')


    def initialize_expression_subclass(self, load_expression_cargo=True, drop_outliers=True):

        [self.minimal_study_parameters.add(i) for i in ['expression_df', 'expression_metadata']]
        self.validate_params()
        #TODO:don't over-write self.expression
        self.expression = ExpressionData(expression_df=self.expression_df,
                                         sample_metadata=self.sample_metadata,
                                         gene_metadata=self.expression_metadata,
                                         load_cargo=load_expression_cargo, drop_outliers=drop_outliers,
                                         species=self.species)
        self.expression.networks = NetworkerViz(self.expression)
        self.default_list_ids.extend(self.expression.lists.keys())

    def initialize_splicing_subclass(self, load_splicing_cargo=False, drop_outliers=True):

        [self.minimal_study_parameters.add(i) for i in ['splicing_df', 'event_metadata']]
        self.validate_params()
        #TODO:don't over-write self.splicing
        self.splicing = SplicingData(splicing=self.splicing_df, sample_metadata=self.sample_metadata,
                                     event_metadata=self.event_metadata,load_cargo=load_splicing_cargo,
                                     drop_outliers=drop_outliers, species=self.species)
        self.splicing.networks = NetworkerViz(self.splicing)

from .._barge_utils import install_development_package

def to_base_file_tuple(tup):
    """for making new packages, auto-loadable data!"""
    assert len(tup) == 2
    return "os.path.join(study_data_dir, %s)" % os.path.basename(tup[0]), tup[1]

new_package_params = ['min_samples','species',
                       'sample_metadata_filename', 'event_metadata_filename', 'expression_data_filename',
                       'splicing_data_filename', 'expression_data_filename', 'gene_metadata_filename',
                       'event_metadata_filename',
                        'default_group_id', 'default_group_ids', 'default_list_id', 'default_list_ids']
class StudyCalls(StudyData):

    def write_package(self, study_name, write_location=None, install=False):
        old_min_params = self.minimal_study_parameters
        write_these = new_package_params
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
        test_package_location = os.path.join(flotilla_install_location, "flotilla_test_project")
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

    def _write_params_file(self, package_location, params_to_write=new_package_params):
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
                        value = to_base_file_tuple(value)

                f.write("%s = %s\n\n" % (param, repr(value)))

    def _add_package_data_resource(self, file_name, data_df, toplevel_package_dir, file_write_mode="tsv"):
        writer = getattr(self, "_write_" + file_write_mode)
        file_base = os.path.basename(file_name)
        rsc_file = os.path.join(toplevel_package_dir, "study_data", file_base + "." + file_write_mode)
        writer(data_df, rsc_file)
        return (rsc_file, file_write_mode)

    def main(self):

        raise NotImplementedError
        #TODO: make this an entry-point, parse flotilla package to load from cmd line, do something
        #this is for the user... who will know little to nothing about queues and the way jobs are done on the backend

        usage = "run_flotilla_cmd cmd_name runner_name"
        # def runner_concept(self, flotilla_package_target = "barebones_package", tool_name):
        #
        #     #a constructor for a new study, takes a long time and maybe runs in parallel. Probably on a cluster...
        #     #same as `import flotilla_package_target as imported_package`
        #
        #     imported_package = __import__(flotilla_package_target)
        #     study = StudyData.__new__()
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
        #TODO: Boyko/Patrick please implement
        raise NotImplementedError


    def compute_jsd(self):
        """Performs Jensen-Shannon Divergence on both splicing and expression study_data

        Jensen-Shannon divergence is a method of quantifying the amount of
        change in distribution of one measurement (e.g. a splicing event or a
        gene expression) from one celltype to another.
        """

        #TODO: Check if JSD has not already been calculated (cacheing or memoizing)

        self.expression.jsd()
        self.splicing.jsd()

        raise NotImplementedError


    def normalize_to_spikein(self):
        raise NotImplementedError

    def compute_expression_splicing_covariance(self):
        raise NotImplementedError


class StudyGraphics(StudyData):
    """
    """
    def __init__(self, *args, **kwargs):

        super(StudyGraphics, self).__init__(*args, **kwargs)
        [self.minimal_study_parameters.add(i) for i in ['expression', 'splicing' ]]

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

class InteractiveStudy(StudyGraphics, StudyCalls):
    """

    Attributes
    ----------


    Methods
    -------

    """
    def __init__(self, *args, **kwargs):
        super(InteractiveStudy, self).__init__(*args, **kwargs)
        self._default_x_pc = 1
        self._default_y_pc = 2
        [self.minimal_study_parameters.add(param) for param in  ['default_group_id', 'default_group_ids',
                                                                    'default_list_id', 'default_list_ids',]]
        [self.minimal_study_parameters.add(i) for i in ['sample_metadata', ]]
        self.validate_params()

    def interactive_pca(self):

        from IPython.html.widgets import interact

        #not sure why nested fxns are required for this, but they are... i think...
        def do_interact(data_type='expression',
                        group_id=self.default_group_id,
                        list_name=self.default_list_id,
                        featurewise=False,
                        list_link='',
                        x_pc=1, y_pc=2,
                        show_point_labels=False,

                        savefile='data/last.pca.pdf'):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                print k, ":", v

            if list_name != "custom" and list_link != "":
                raise ValueError("set list_name to \"custom\" to use list_link")

            if list_name == "custom" and list_link == "":
                raise ValueError("use a custom list name please")

            if list_name == 'custom':
                list_name = list_link

            self.plot_pca(group_id=group_id, data_type=data_type,
                     featurewise=featurewise,
                     x_pc=x_pc, y_pc=y_pc, show_point_labels=show_point_labels,
                     list_name=list_name)
            if savefile != '':
                f = plt.gcf()
                f.savefig(savefile)

        interact(do_interact,
                 data_type=('expression', 'splicing'),
                 group_id=self.default_group_ids,
                 list_name=self.default_list_ids + ["custom"],
                 featurewise=False,
                 x_pc=(1, 10), y_pc=(1, 10),
                 show_point_labels=False, )


    def interactive_graph(self):
        from IPython.html.widgets import interact

        #not sure why nested fxns are required for this, but they are... i think...
        def do_interact(data_type='expression', group_id=self.default_group_id,
                        list_name=self.default_list_id,weight_fun=NetworkerViz.weight_funs,
                        featurewise=False,
                        use_pc_1=True, use_pc_2=True, use_pc_3=True,
                        use_pc_4=True,degree_cut=1,
                        cov_std_cut=1.8, n_pcs=5,
                        feature_of_interest="RBFOX2",
                        draw_labels=False,
                        savefile='data/last.graph.pdf',
                        ):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                print k, ":", v

            if data_type == 'expression':
                assert (list_name in self.expression.lists.keys())
            if data_type == 'splicing':
                assert (list_name in self.expression.lists.keys())

            self.plot_graph(data_type=data_type,
                            group_id=group_id,
                            list_name=list_name,

                       featurewise=featurewise, draw_labels=draw_labels,
                       degree_cut=degree_cut, cov_std_cut=cov_std_cut,
                       n_pcs=n_pcs,
                       feature_of_interest=feature_of_interest,
                       use_pc_1=use_pc_1, use_pc_2=use_pc_2, use_pc_3=use_pc_3,
                       use_pc_4=use_pc_4,
                       wt_fun=weight_fun)
            if savefile is not '':
                plt.gcf().savefig(savefile)

        all_lists = list(
            set(self.expression.lists.keys() + self.splicing.lists.keys()))
        interact(do_interact,
                 data_type=('expression', 'splicing'),
                 group_id=self.default_group_ids,
                 list_name=all_lists,
                 featurewise=False,
                 cov_std_cut=(0.1, 3),
                 degree_cut=(0, 10),
                 n_pcs=(2, 100),
                 draw_labels=False,
                 feature_of_interest="RBFOX2",
                 use_pc_1=True, use_pc_2=True, use_pc_3=True, use_pc_4=True,
        )

    def interactive_classifier(self):

        from IPython.html.widgets import interact

        #not sure why nested fxns are required for this, but they are... i think...
        def do_interact(data_type='expression',
                        group_id=self.default_group_id,
                        list_name=self.default_list_id,
                        categorical_variable='outlier',
                        feature_score_std_cutoff=2,
                        savefile='data/last.clf.pdf'):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                print k, ":", v

            if data_type == 'expression':
                data_obj = self.expression
            if data_type == 'splicing':
                data_obj = self.splicing

            assert (list_name in data_obj.lists.keys())

            prd = data_obj.get_classifier(list_name, group_id,
                                         categorical_variable)
            prd(categorical_variable,
                feature_score_std_cutoff=feature_score_std_cutoff)
            print "retrieve this classifier with:\nprd=study.%s.get_predictor('%s', '%s', '%s')\n\
pca=prd('%s', feature_score_std_cutoff=%f)" \
                  % (data_type, list_name, group_id, categorical_variable,
                     categorical_variable, feature_score_std_cutoff)
            if savefile is not '':
                plt.gcf().savefig(savefile)

        all_lists = list(
            set(self.expression.lists.keys() + self.splicing.lists.keys()))
        interact(do_interact,
                 data_type=('expression', 'splicing'),
                 group_id=self.default_group_ids,
                 list_name=all_lists,
                 categorical_variable=[i for i in self.default_group_ids if
                                       not i.startswith("~")],
                 feature_score_std_cutoff=(0.1, 20),
                 draw_labels=False,
        )

    def interactive_localZ(self):

        from IPython.html.widgets import interact

        def do_interact(data_type='expression', sample1='', sample2='',
                        pCut='0.01'):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                print k, ":", v
            pCut = float(pCut)
            assert pCut > 0
            if data_type == 'expression':
                data_obj = self.expression
            if data_type == 'splicing':
                data_obj = self.splicing

            try:
                assert sample1 in data_obj.df.index
            except:
                print "sample: %s, is not in %s DataFrame, try a different sample ID" % (
                sample1, data_type)
                return
            try:
                assert sample2 in data_obj.df.index
            except:
                print "sample: %s, is not in %s DataFrame, try a different sample ID" % (
                sample2, data_type)
                return
            self.localZ_result = data_obj.twoway(sample1, sample2,
                                                 pCut=pCut).result_
            print "localZ finished, find the result in <this_obj>.localZ_result_"

        interact(do_interact,
                 data_type=('expression', 'splicing'),
                 sample1='replaceme',
                 sample2='replaceme',
                 pCut='0.01')
