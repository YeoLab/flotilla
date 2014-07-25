"""
Data models for "studies" studies include attributes about the data and are
heavier in terms of data load
"""

from collections import defaultdict
import json
import os
import sys
import warnings

import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metadata import MetaData
from .expression import ExpressionData, SpikeInData
from .quality_control import MappingStatsData
from .splicing import SplicingData, FRACTION_DIFF_THRESH
from ..visualize.color import blue
from ..visualize.ipython_interact import Interactive
from ..visualize.network import NetworkerViz
from ..external import data_package_url_to_dict, check_if_already_downloaded


SPECIES_DATA_PACKAGE_BASE_URL = 'http://sauron.ucsd.edu/flotilla_projects'

# import flotilla
# FLOTILLA_DIR = os.path.dirname(flotilla.__file__)


class StudyFactory(object):
    _accepted_filetypes = 'tsv'

    def __init__(self):
        self.minimal_study_parameters = set()
        self.new_study_params = set()
        self.getters = []
        self.default_sample_subset = None
        self.default_feature_subset = None

    def __setattr__(self, key, value):
        """Check if the attribute already exists and warns on overwrite.
        """
        if hasattr(self, key):
            warnings.warn('Over-writing attribute {}'.format(key))
        super(StudyFactory, self).__setattr__(key, value)

    @staticmethod
    def _to_base_file_tuple(tup):
        """for making new packages, auto-loadable data!"""
        assert len(tup) == 2
        return "os.path.join(study_data_dir, %s)" % os.path.basename(tup[0]), \
               tup[1]

    def _add_package_data_resource(self, file_name, data_df,
                                   toplevel_package_dir,
                                   file_write_mode="tsv"):
        writer = getattr(self, "_write_" + file_write_mode)
        file_base = os.path.basename(file_name)
        rsc_file = os.path.join(toplevel_package_dir, "study_data",
                                file_base + "." + file_write_mode)
        writer(data_df, rsc_file)
        return (rsc_file, file_write_mode)

    def validate_params(self):
        """make sure that all necessary attributes are present"""
        for param in self.minimal_study_parameters:
            try:
                x = getattr(self, param)
            except KeyError:
                raise AssertionError("Missing minimal parameter %s" % param)

    @staticmethod
    def _load_pickle_df(file_name):
        return pd.read_pickle(file_name)

    @staticmethod
    def _write_pickle_df(df, file_name):
        df.to_pickle(file_name)

    @staticmethod
    def _load_gzip_pickle_df(file_name):
        import cPickle
        import gzip

        with gzip.open(file_name, 'r') as f:
            return cPickle.load(f)

    @staticmethod
    def _write_gzip_pickle_df(df, file_name):
        import tempfile

        tmpfile_h, tmpfile = tempfile.mkstemp()
        df.to_pickle(tmpfile)
        import subprocess

        subprocess.call(['gzip -f %s' % tempfile])
        subprocess.call(['mv %s %s' % (tempfile, file_name)])

    @staticmethod
    def _load_tsv(file_name, compression=None):
        return pd.read_table(file_name, index_col=0, compression=compression)

    @staticmethod
    def _load_json(filename, compression=None):
        """
        Parameters
        ----------
        filename : str
            Name of the json file toread
        compression : str
            Not used, only for  compatibility with other load functions

        Returns
        -------


        Raises
        ------
        """
        return pd.read_json(filename)

    @staticmethod
    def _write_tsv(df, file_name):
        df.to_csv(file_name, sep='\t')

    @staticmethod
    def _load_csv(file_name, compression=None):
        return pd.read_csv(file_name, index_col=0, compression=compression)

    @staticmethod
    def _write_csv(df, file_name):
        df.to_csv(file_name)

    def _get_loading_method(self, file_name):
        """loading_methods for loading from file"""
        return getattr(self, "_load_" + file_name)

    def load(self, file_name, file_type='pickle_df'):
        return self._get_loading_method(file_type)(file_name)


class Study(StudyFactory):
    """A biological study, with associated metadata, expression, and splicing
    data.
    """
    default_feature_set_ids = []

    # Data types with enough data that we'd probably reduce them, and even
    # then we might want to take subsets. E.g. most variant genes for
    # expresion. But we don't expect to do this for spikein or mapping_stats
    # data
    _subsetable_data_types = ['expression', 'splicing']

    initializers = {'metadata_data': MetaData,
                    'expression_data': ExpressionData,
                    'splicing_data': SplicingData,
                    'mapping_stats_data': MappingStatsData,
                    'spikein_data': SpikeInData}

    readers = {'tsv': StudyFactory._load_tsv,
               'csv': StudyFactory._load_csv,
               'json': StudyFactory._load_json,
               'pickle_df': StudyFactory._load_pickle_df,
               'gzip_pickle_df': StudyFactory._load_gzip_pickle_df}

    _default_reducer_kwargs = {'whiten': False,
                               'show_point_labels': False,
                               'show_vectors': False}

    _default_plot_kwargs = {'marker': 'o', 'color': blue}

    def __init__(self, sample_metadata, expression_data=None,
                 splicing_data=None,
                 expression_feature_data=None,
                 expression_feature_rename_col='gene_name',
                 splicing_feature_data=None,
                 splicing_feature_rename_col='gene_name',
                 mapping_stats_data=None,
                 mapping_stats_number_mapped_col="Uniquely mapped reads "
                                                 "number",
                 spikein_data=None,
                 spikein_feature_data=None,
                 drop_outliers=True, species=None,
                 gene_ontology_data=None,
                 expression_log_base=None,
                 metadata_pooled_col=None,
                 phenotype_order=None,
                 phenotype_to_color=None,
                 phenotype_to_marker=None):
        """Construct a biological study

        This class only accepts data, no filenames. All data must already
        have been read in and exist as Python objects.

        Parameters
        ----------
        #TODO: Maybe make these all kwargs?
        sample_metadata : pandas.DataFrame
            Only required parameter. Samples as the index, with features as
            columns. If there is a column named "color", this will be used as
            the color for that sample in PCA and other plots. If there is no
            color but there is a column named "celltype", then colors for
            each of the different celltypes will be auto-created.
        expression_data : pandas.DataFrame
            Samples x feature dataframe of gene expression measurements,
            e.g. from an RNA-Seq or a microarray experiment. Assumed to be
            log-normal (i.e. not log-transformed)
        expression_feature_data : pandas.DatFrame
            features x other_features dataframe describing other parameters
            of the gene expression features, e.g. mapping Ensembl IDs to gene
            symbols or gene biotypes.
        expression_feature_rename_col : str
            A column name in the expression_feature_data dataframe that you'd
            like to rename the expression features to, in the plots. For
            example, if your gene IDs are Ensembl IDs, but you want to plot
            UCSC IDs, make sure the column you want, e.g. "ucsc_id" is in your
            dataframe and specify that. Default "gene_name"
        splicing_data : pandas.DataFrame
            Samples x feature dataframe of percent spliced in scores, e.g. as
            measured by the program MISO. Assumed that these values only fall
            between 0 and 1.
        splicing_feature_data : pandas.DataFrame
            features x other_features dataframe describing other parameters
            of the splicing features, e.g. mapping MISO IDs to Ensembl IDs or
            gene symbols or transcript types
        splicing_feature_rename_col : str
            A column name in the splicing_feature_data dataframe that you'd
            like to rename the splicing features to, in the plots. For
            example, if your splicing IDs are MISO IDs, but you want to plot
            Ensembl IDs, make sure the column you want, e.g. "ensembl_id" is
            in your dataframe and specify that. Default "gene_name".
        mapping_stats_data : pandas.DataFrame
            Samples x feature dataframe of mapping stats measurements.
            Currently, this
        mapping_stats_number_mapped_col : str
            A column name in the mapping_stats_data which specifies the
            number of (uniquely or not) mapped reads. Default "Uniquely
            mapped reads number"
        spikein_data : pandas.DataFrame
            samples x features DataFrame of spike-in expression values
        spikein_feature_data : pandas.DataFrame
            Features x other_features dataframe, e.g. of the molecular
            concentration of particular spikein transcripts
        drop_outliers : bool
            Whether or not to drop samples indicated as outliers in the
            sample_metadata from the other data, i.e. with a column
            named 'outlier' in sample_metadata, then remove those
            samples from expression_data for further analysis
        species : str
            Name of the species and genome version, e.g. 'hg19' or 'mm10'.
        gene_ontology_data : pandas.DataFrame
            Gene ids x ontology categories dataframe used for GO analysis.
        metadata_pooled_col : str
            Column in metadata_data which specifies as a boolean
            whether or not this sample was pooled.

        Note
        ----
        This function explicitly specifies ALL the instance variables (except
        those that are marked by the @property decorator), because,
        as described [1], "If you write initialization functions separate from
        __init__ then experienced developers will certainly see your code as a
        kid's playground."

        [1] http://stackoverflow.com/q/12513185/1628971
        """
        super(Study, self).__init__()
        # if params_dict is None:
        #     params_dict = {}
        # self.update(params_dict)
        # self.initialize_required_getters()
        # self.apply_getters()
        self.species = species
        self.gene_ontology_data = gene_ontology_data

        self.metadata = MetaData(sample_metadata, phenotype_order,
                                 phenotype_to_color, phenotype_to_marker)
        self.phenotype_order = self.metadata.phenotype_order
        self.phenotype_to_color = self.metadata.phenotype_to_color
        self.phenotype_to_marker = self.metadata.phenotype_to_marker

        self.default_sample_subsets = \
            [col for col in self.metadata.data.columns
             if self.metadata.data[col].dtype == bool]
        self.default_sample_subsets.insert(0, 'all_samples')

        if 'outlier' in self.metadata.data and drop_outliers:
            outliers = self.metadata.data.index[
                self.metadata.data.outlier.astype(bool)]
        else:
            outliers = None
            self.metadata.data['outlier'] = False

        # Get pooled samples
        if metadata_pooled_col is not None:
            if metadata_pooled_col in self.metadata.data:
                try:
                    pooled = self.metadata.data.index[
                        self.metadata.data[
                            metadata_pooled_col].astype(bool)]
                except:
                    pooled = None
        else:
            pooled = None

        if expression_data is not None:
            self.expression = ExpressionData(
                expression_data,
                expression_feature_data,
                feature_rename_col=expression_feature_rename_col,
                outliers=outliers,
                log_base=expression_log_base, pooled=pooled)
            self.expression.networks = NetworkerViz(self.expression)
            self.default_feature_set_ids.extend(self.expression.feature_subsets
                                                .keys())
        if splicing_data is not None:
            self.splicing = SplicingData(
                splicing_data, splicing_feature_data,
                feature_rename_col=splicing_feature_rename_col,
                outliers=outliers, pooled=pooled)
            self.splicing.networks = NetworkerViz(self.splicing)
        if mapping_stats_data is not None:
            self.mapping_stats = MappingStatsData(
                mapping_stats_data,
                mapping_stats_number_mapped_col)
        if spikein_data is not None:
            self.spikein = SpikeInData(spikein_data, spikein_feature_data)
        sys.stderr.write("subclasses initialized\n")
        self.validate_params()
        sys.stderr.write("package validated\n")
        # try:
        #     self.hack_lazy()
        # except:
        #     print "hackery failed"

    @property
    def phenotype_color_order(self):
        return [self.phenotype_to_color[p] for p in self.phenotype_order]

    @property
    def default_feature_subsets(self):
        feature_subsets = {}
        for name in self._subsetable_data_types:
            try:
                data_type = getattr(self, name)
            except AttributeError:
                continue
            feature_subsets[name] = data_type.feature_subsets
        return feature_subsets

    @property
    def sample_id_to_color(self):
        """If "color" is a column in the metadata data, return a
        dict of that {sample_id: color} mapping, else try to create it using
        the "celltype" columns, else just return a dict mapping to a default
        color (blue)
        """
        if 'color' in self.metadata.data:
            return self.metadata.data.color.to_dict()
        elif 'celltype' in self.metadata.data:
            grouped = self.metadata.data.groupby('celltype')
            palette = iter(sns.color_palette(n_colors=grouped.ngroups + 1))
            color = grouped.apply(lambda x: mplcolors.rgb2hex(palette.next()))
            color.name = 'color'
            self.metadata.data = self.metadata.data.join(
                color, on='celltype')

            return self.metadata.data.color.to_dict()
        else:
            return defaultdict(lambda x: blue)

    @property
    def sample_id_to_celltype(self):
        """If "celltype" is a column in the metadata data, return a
        dict of that {sample_id: celltype} mapping.
        """
        if 'celltype' in self.metadata.data:
            return self.metadata.data.celltype
        else:
            return pd.Series(['celltype'],
                             index=self.metadata.data.index)

    @classmethod
    def from_data_package_url(
            cls, data_package_url,
            species_data_package_base_url=SPECIES_DATA_PACKAGE_BASE_URL):
        """Create a study from a url of a datapackage.json file

        Parameters
        ----------
        data_package_url : str
            HTTP url of a datapackage.json file, following the specification
            described here: http://dataprotocols.org/data-packages/ and
            requiring the following data resources: metadata,
            expression, splicing
        species_data_pacakge_base_url : str
            Base URL to fetch species-specific gene and splicing event
            metadata from. Default 'http://sauron.ucsd.edu/flotilla_projects'

        Returns
        -------
        study : Study
            A "study" object containing the data described in the
            data_package_url file

        Raises
        ------
        AttributeError
            If the datapackage.json file does not contain the required
            resources of metadata, expression, and splicing.
        """
        data_package = data_package_url_to_dict(data_package_url)
        return cls.from_data_package(
            data_package,
            species_data_package_base_url=species_data_package_base_url)

    @classmethod
    def from_data_package_file(
            cls, data_package_filename,
            species_data_package_base_url=SPECIES_DATA_PACKAGE_BASE_URL):
        with open(data_package_filename) as f:
            data_package = json.load(f)
        return cls.from_data_package(data_package,
                                     species_data_package_base_url)

    @classmethod
    def from_data_package(
            cls, data_package,
            species_data_package_base_url=SPECIES_DATA_PACKAGE_BASE_URL):
        dfs = {}
        log_base = None
        metadata_pooled_col = None
        phenotype_order = None
        phenotype_to_color = None
        phenotype_to_marker = None

        for resource in data_package['resources']:
            if 'url' in resource:
                resource_url = resource['url']
                filename = check_if_already_downloaded(resource_url)
            else:
                filename = resource['path']

            name = resource['name']

            reader = cls.readers[resource['format']]
            compression = None if 'compression' not in resource else \
                resource['compression']

            dfs[name] = reader(filename, compression=compression)

            if name == 'expression':
                if 'log_transformed' in resource:
                    log_base = 2
            if name == 'metadata':
                if 'pooled_col' in resource:
                    metadata_pooled_col = resource['pooled_col']
                if 'phenotype_order' in resource:
                    phenotype_order = resource['phenotype_order']
                if 'phenotype_to_color' in resource:
                    phenotype_to_color = resource['phenotype_to_color']
                if 'phenotype_to_marker' in resource:
                    phenotype_to_marker = resource['phenotype_to_marker']

        if 'species' in data_package:
            species_data_url = '{}/{}/datapackage.json'.format(
                species_data_package_base_url, data_package['species'])
            species_data_package = data_package_url_to_dict(species_data_url)
            species_dfs = {}

            for resource in species_data_package['resources']:
                if 'url' in resource:
                    resource_url = resource['url']
                    filename = check_if_already_downloaded(resource_url)
                else:
                    filename = resource['path']

                # reader = getattr(cls, '_load_' + resource['format'])
                reader = cls.readers[resource['format']]

                compression = None if 'compression' not in resource else \
                    resource['compression']
                name = resource['name']
                species_dfs[name] = reader(filename,
                                           compression=compression)
                if 'feature_rename_col' in resource:
                    key = '{}_feature_rename_col'.format(
                        name.split('_feature_data')[0])
                    species_dfs[key] = resource['feature_rename_col']
        else:
            species_dfs = {}

        try:
            sample_metadata = dfs['metadata']
            expression_data = None if 'expression' not in dfs else dfs[
                'expression']
            splicing_data = None if 'splicing' not in dfs else dfs['splicing']
        except KeyError:
            raise AttributeError('The datapackage.json file is required to '
                                 'have the "metadata" resource')
        try:
            mapping_stats_data = dfs['mapping_stats']
        except KeyError:
            mapping_stats_data = None
        try:
            spikein_data = dfs['spikein']
        except KeyError:
            spikein_data = None

        study = Study(
            sample_metadata=sample_metadata,
            expression_data=expression_data,
            splicing_data=splicing_data,
            mapping_stats_data=mapping_stats_data,
            spikein_data=spikein_data,
            # expression_feature_rename_col='gene_name',
            # splicing_feature_rename_col='gene_name',
            expression_log_base=log_base,
            metadata_pooled_col=metadata_pooled_col,
            phenotype_order=phenotype_order,
            phenotype_to_color=phenotype_to_color,
            phenotype_to_marker=phenotype_to_marker,
            **species_dfs)
        return study

    def __add__(self, other):
        """Sanely concatenate one or more Study objects
        """
        raise NotImplementedError
        self.metadata = MetaData(
            pd.concat([self.metadata.data,
                       other.metadata.data]))
        self.expression.data = ExpressionData(
            pd.concat([self.expression.data,
                       other.expression.data]))

    def _set_plot_colors(self):
        """If there is a column 'color' in the sample metadata, specify this
        as the plotting color
        """
        try:
            self._default_reducer_kwargs.update(
                {'colors_dict': self.metadata_data.color})
            self._default_plot_kwargs.update(
                {'color': self.metadata_data.color.tolist()})
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
                {'markers_dict': self.metadata_data.marker})
            self._default_plot_kwargs.update(
                {'marker': self.metadata_data.marker.tolist()})
        except AttributeError:
            sys.stderr.write("There is no column named 'marker' in the sample "
                             "metadata, defaulting to a circle for all "
                             "samples\n")
            self._default_reducer_kwargs.update(
                {'markers_dict': defaultdict(lambda: 'o')})

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
        # TODO.md: Boyko/Patrick please implement
        raise NotImplementedError

    def jsd(self):
        """Performs Jensen-Shannon Divergence on both splicing and expression
        study_data

        Jensen-Shannon divergence is a method of quantifying the amount of
        change in distribution of one measurement (e.g. a splicing event or a
        gene expression) from one celltype to another.
        """
        raise NotImplementedError
        # TODO: Check if JSD has not already been calculated (memoize)
        self.expression.jsd()
        self.splicing.jsd()

    def normalize_to_spikein(self):
        raise NotImplementedError

    def compute_expression_splicing_covariance(self):
        raise NotImplementedError

    @staticmethod
    def maybe_make_directory(filename):
        # Make the directory if it's not already there
        try:
            directory = os.path.abspath(os.path.dirname(filename))
            os.makedirs(os.path.abspath(directory))
        except OSError:
            pass

    def feature_subset_to_feature_ids(self, data_type, feature_subset=None,
                                      rename=False):
        """Given a name of a feature subset, get the associated feature ids

        Parameters
        ----------
        data_type : str
            A string describing the data type, e.g. "expression"
        feature_subset : str
            A string describing the subset of data type (must be already
            calculated)

        Returns
        -------
        feature_ids : list of strings
            List of features ids from the specified datatype
        """
        if 'expression'.startswith(data_type):
            return self.expression.feature_subset_to_feature_ids(
                feature_subset, rename)
        elif 'splicing'.startswith(data_type):
            return self.splicing.feature_subset_to_feature_ids(
                feature_subset, rename)

    def sample_subset_to_sample_ids(self, phenotype_subset=None):
        """Convert a string naming a subset of phenotypes in the data into
        sample ids

        Parameters
        ----------
        phenotype_subset : str
            A valid string describing a boolean phenotype described in the
            metadata data

        Returns
        -------
        sample_ids : list of strings
            List of sample ids in the data
        """
        if phenotype_subset is None or 'all_samples'.startswith(
                phenotype_subset):
            sample_ind = np.ones(self.metadata.data.shape[0],
                                 dtype=bool)
        elif phenotype_subset.startswith("~"):
            sample_ind = ~pd.Series(
                self.metadata.data[phenotype_subset.lstrip("~")],
                dtype='bool')
        else:
            sample_ind = pd.Series(
                self.metadata.data[phenotype_subset], dtype='bool')
        sample_ids = self.metadata.data.index[sample_ind]
        return sample_ids

    def plot_pca(self, data_type='expression', x_pc=1, y_pc=2,
                 sample_subset=None, feature_subset=None,
                 title='', featurewise=False,
                 show_point_labels=False,
                 **kwargs):
        """Performs PCA on both expression and splicing study_data

        Parameters
        ----------
        data_type : str
            One of the names of the data types, e.g. "expression" or
            "splicing"
        x_pc : int
            Which principal component to plot on the x-axis
        y_pc : int
            Which principal component to plot on the y-axis
        sample_subset : str or None
            Which subset of the samples to use, based on some phenotype
            column in the experiment design data. If None, all samples are
            used.
        feature_subset : str or None
            Which subset of the features to used, based on some feature type
            in the expression data (e.g. "variant"). If None, all features
            are used.
        title : str
            The title of the plot
        show_point_labels : bool
            Whether or not to show the labels of the points. If this is
            samplewise (default), then this labels the samples. If this is
            featurewise, then this labels the features.

        Raises
        ------

        """
        sample_ids = self.sample_subset_to_sample_ids(sample_subset)
        feature_ids = self.feature_subset_to_feature_ids(data_type,
                                                         feature_subset,
                                                         rename=False)
        # TODO: move this kwarg stuff into visualize
        kwargs['x_pc'] = x_pc
        kwargs['y_pc'] = y_pc
        kwargs['sample_ids'] = sample_ids
        kwargs['feature_ids'] = feature_ids
        kwargs['title'] = title
        kwargs['featurewise'] = featurewise
        kwargs['show_point_labels'] = show_point_labels
        kwargs['groupby'] = None
        if not featurewise:
            kwargs['label_to_color'] = self.phenotype_to_color
            kwargs['label_to_marker'] = self.phenotype_to_marker
            kwargs['groupby'] = self.sample_id_to_celltype

        if "expression".startswith(data_type):
            reducer = self.expression.plot_pca(**kwargs)
        elif "splicing".startswith(data_type):
            reducer = self.splicing.plot_pca(**kwargs)
        else:
            raise ValueError('The data type {} does not exist in this study'
                             .format(data_type))
        return reducer

    def plot_graph(self, data_type='expression', sample_subset=None,
                   feature_subset=None,
                   **kwargs):
        """Plot the graph (network) of these data

        Parameters
        ----------
        data_type : str
            One of the names of the data types, e.g. "expression" or "splicing"
        sample_subset : str or None
            Which subset of the samples to use, based on some phenotype
            column in the experiment design data. If None, all samples are
            used.
        feature_subset : str or None
            Which subset of the features to used, based on some feature type
            in the expression data (e.g. "variant"). If None, all features
            are used.
        """
        sample_ids = self.sample_subset_to_sample_ids(sample_subset)
        feature_ids = self.feature_subset_to_feature_ids(data_type,
                                                         feature_subset,
                                                         rename=False)

        kwargs['sample_id_to_color'] = self.sample_id_to_color
        kwargs['sample_ids'] = sample_ids
        kwargs['feature_ids'] = feature_ids

        if data_type == "expression":
            self.expression.networks.draw_graph(**kwargs)
        elif data_type == "splicing":
            self.splicing.networks.draw_graph(**kwargs)

    def plot_study_sample_legend(self):
        markers = self.metadata.data.color.groupby(
            self.metadata.data.marker
            + "." + self.metadata.data.celltype).last()

        f, ax = plt.subplots(1, 1, figsize=(3, len(markers)))

        for i, point_type in enumerate(markers.iteritems(), ):
            mrk, celltype = point_type[0].split('.')
            ax.scatter(0, 0, marker=mrk, c=point_type[1],
                       edgecolor='none', label=celltype,
                       s=160)
        ax.set_xlim(1, 2)
        ax.set_ylim(1, 2)
        ax.axis('off')
        legend = ax.legend(title='cell type', fontsize=20, )
        return legend

    def plot_classifier(self, trait, data_type='expression', title='',
                        show_point_labels=False, sample_subset=None,
                        feature_subset=None,
                        **kwargs):
        """Plot a predictor for the specified data type and trait(s)

        Parameters
        ----------
        data_type : str
            One of the names of the data types, e.g. "expression" or "splicing"
        trait : str
            Column name in the metadata data that you would like
            to classify on

        Returns
        -------


        """
        trait_data = self.metadata.data[trait]
        sample_ids = self.sample_subset_to_sample_ids(sample_subset)
        feature_ids = self.feature_subset_to_feature_ids(data_type,
                                                         feature_subset,
                                                         rename=False)

        kwargs['trait'] = trait_data
        kwargs['title'] = title
        kwargs['show_point_labels'] = show_point_labels
        kwargs['colors_dict'] = self.sample_id_to_color
        kwargs['sample_ids'] = sample_ids
        kwargs['feature_ids'] = feature_ids
        # print(kwargs.keys())

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

    def modalities(self, sample_subset=None, feature_subset=None):
        """Get modality assignments of

        """
        sample_ids = self.sample_subset_to_sample_ids(sample_subset)
        feature_ids = self.feature_subset_to_feature_ids('splicing',
                                                         feature_subset,
                                                         rename=False)
        return self.splicing.modalities(sample_ids, feature_ids)

    def plot_modalities(self, sample_subset=None, feature_subset=None,
                        normed=True):
        # try:
        sample_ids = self.sample_subset_to_sample_ids(sample_subset)
        feature_ids = self.feature_subset_to_feature_ids('splicing',
                                                         feature_subset,
                                                         rename=False)

        grouped = self.sample_id_to_celltype.groupby(
            self.sample_id_to_celltype)

        # Account for bar plot and plot of the reduced space of ALL samples
        n = grouped.ngroups + 2
        groups = ['all']
        fig, axes = plt.subplots(ncols=n, figsize=(n * 4, 4))
        bar_ax = axes[0]
        all_ax = axes[1]
        self.splicing.plot_modalities_reduced(sample_ids, feature_ids,
                                              all_ax, title='all samples')
        self.splicing.plot_modalities_bar(sample_ids, feature_ids,
                                          bar_ax, i=0, normed=normed,
                                          legend=False)

        axes = axes[2:]
        for i, ((celltype, series), ax) in enumerate(zip(grouped, axes)):
            groups.append(celltype)
            sys.stdout.write('\n---- {} ----\n'.format(celltype))
            samples = series.index.intersection(sample_ids)
            # legend = i == 0
            self.splicing.plot_modalities_bar(samples, feature_ids,
                                              bar_ax, i + 1, normed=normed,
                                              legend=False)

            self.splicing.plot_modalities_reduced(samples, feature_ids,
                                                  ax, title=celltype)

        bar_ax.set_xticks(np.arange(len(groups)) + 0.4)
        bar_ax.set_xticklabels(groups)
        # except AttributeError:
        #     pass

    def celltype_sizes(self, data_type='splicing'):
        if data_type == 'expression':
            self.expression.data.groupby(self.sample_id_to_celltype,
                                         axis=0).size()
        if data_type == 'splicing':
            self.splicing.data.groupby(self.sample_id_to_celltype,
                                       axis=0).size()

    @property
    def celltype_event_counts(self):
        """Number of cells that detected this event in that celltype
        """
        return self.splicing.data.groupby(
            self.sample_id_to_celltype, axis=0).apply(
            lambda x: x.groupby(level=0, axis=0).transform(
                lambda x: x.count()).sum()).replace(0, np.nan)

    def unique_celltype_event_counts(self, n=1):
        celltype_event_counts = self.celltype_event_counts
        return celltype_event_counts[celltype_event_counts <= n]

    def percent_unique_celltype_events(self, n=1):
        n_unique = self.unique_celltype_event_counts(n).sum(axis=1)
        n_total = self.celltype_event_counts.sum(axis=1).astype(float)
        return n_unique / n_total * 100

    @property
    def celltype_modalities(self):
        """Return modality assignments of each celltype
        """
        return self.splicing.data.groupby(
            self.sample_id_to_celltype, axis=0).apply(
            lambda x: self.splicing.modalities(x.index))

    def plot_modalities_lavalamps(self, sample_subset=None, bootstrapped=False,
                                  bootstrapped_kws=None):
        grouped = self.splicing.data.groupby(self.sample_id_to_color, axis=0)
        celltype_groups = self.splicing.data.groupby(
            self.sample_id_to_celltype, axis=0)

        if sample_subset is not None:
            # Only plotting one sample_subset, use the modality assignments
            # from just the samples from this sample_subset
            celltype_samples = celltype_groups.groups[sample_subset]
            celltype_samples = set(celltype_groups.groups[sample_subset])
            use_these_modalities = True
        else:
            # Plotting all the celltypes, use the modality assignments from
            # all celltypes together
            celltype_samples = self.splicing.data.index
            use_these_modalities = False

        for i, (color, sample_ids) in enumerate(grouped.groups.iteritems()):
            x_offset = 1. / (i + 1)
            sample_ids = celltype_samples.intersection(sample_ids)
            if len(sample_ids) > 0:
                self.splicing.plot_modalities_lavalamps(
                    sample_ids=sample_ids,
                    color=color,
                    x_offset=x_offset,
                    use_these_modalities=use_these_modalities,
                    bootstrapped=bootstrapped,
                    bootstrapped_kws=bootstrapped_kws)

    def plot_event(self, feature_id, sample_subset=None, ax=None):
        """Plot the violinplot of an event

        """
        sample_ids = self.sample_subset_to_sample_ids(sample_subset)
        self.splicing.plot_event(feature_id, sample_ids,
                                 phenotype_groupby=self.sample_id_to_celltype,
                                 ax=ax)

    def plot_lavalamp_pooled_inconsistent(
            self, celltype=None, feature_ids=None,
            fraction_diff_thresh=FRACTION_DIFF_THRESH):
        # grouped_ids = self.splicing.data.groupby(self.sample_id_to_color,
        #                                          axis=0)
        celltype_groups = self.metadata.data.groupby(
            self.sample_id_to_celltype, axis=0)

        if celltype is not None:
            # Only plotting one celltype
            celltype_samples = set(celltype_groups.groups[celltype])
        else:
            # Plotting all the celltypes
            celltype_samples = self.metadata.data.index

        celltype_and_sample_ids = celltype_groups.groups.iteritems()
        for i, (celltype, sample_ids) in enumerate(celltype_and_sample_ids):
            # import pdb; pdb.set_trace()

            # Assumes all samples of a celltype have the same color...
            # probably wrong
            color = self.sample_id_to_color[sample_ids[0]]
            sample_ids = celltype_samples.intersection(sample_ids)
            if len(sample_ids) == 0:
                continue
            self.splicing.plot_lavalamp_pooled_inconsistent(
                sample_ids, feature_ids, fraction_diff_thresh,
                color=color)

    def percent_pooled_inconsistent(self,
                                    celltype=None, feature_ids=None,
                                    fraction_diff_thresh=FRACTION_DIFF_THRESH):

        celltype_groups = self.metadata.data.groupby(
            self.sample_id_to_celltype, axis=0)

        if celltype is not None:
            # Only plotting one celltype
            celltype_samples = set(celltype_groups.groups[celltype])
        else:
            # Plotting all the celltypes
            celltype_samples = self.metadata.data.index

        celltype_and_sample_ids = celltype_groups.groups.iteritems()
        for i, (celltype, sample_ids) in enumerate(celltype_and_sample_ids):
            # import pdb; pdb.set_trace()

            # Assumes all samples of a celltype have the same color...
            # probably wrong
            color = self.sample_id_to_color[sample_ids[0]]
            sample_ids = celltype_samples.intersection(sample_ids)
            if len(sample_ids) == 0:
                continue
            self.splicing.percent_pooled_inconsistent(sample_ids, feature_ids,
                                                      fraction_diff_thresh)

    def plot_clusteredheatmap(self, sample_subset=None,
                              feature_subset='variant',
                              data_type='expression', metric='euclidean',
                              linkage_method='median'):
        if data_type == 'expression':
            data = self.expression.data
        elif data_type == 'splicing':
            data = self.splicing.data
        celltype_groups = data.groupby(
            self.sample_id_to_celltype, axis=0)

        if sample_subset is not None:
            # Only plotting one sample_subset
            try:
                sample_ids = set(celltype_groups.groups[sample_subset])
            except KeyError:
                sample_ids = self.sample_subset_to_sample_ids(sample_subset)
        else:
            # Plotting all the celltypes
            sample_ids = data.index

        sample_colors = [self.sample_id_to_color[x] for x in sample_ids]
        feature_ids = self.feature_subset_to_feature_ids(data_type,
                                                         feature_subset,
                                                         rename=False)

        if data_type == "expression":
            return self.expression.plot_clusteredheatmap(
                sample_ids, feature_ids, linkage_method=linkage_method,
                metric=metric, sample_colors=sample_colors)
        elif data_type == "splicing":
            return self.splicing.plot_clusteredheatmap(
                sample_ids, feature_ids, linkage_method=linkage_method,
                metric=metric, sample_colors=sample_colors)

    def plot_event(self, feature_id, sample_ids=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        self.splicing.plot_event(feature_id, sample_ids,
                                 phenotype_groupby=self.sample_id_to_celltype,
                                 phenotype_order=self.metadata.phenotype_order,
                                 color=self.phenotype_color_order, ax=ax)


# Add interactive visualizations
Study.interactive_classifier = Interactive.interactive_classifier
Study.interactive_graph = Interactive.interactive_graph
Study.interactive_pca = Interactive.interactive_pca
Study.interactive_localZ = Interactive.interactive_localZ
Study.interactive_lavalamp_pooled_inconsistent = \
    Interactive.interactive_lavalamp_pooled_inconsistent
Study.interactive_clusteredheatmap = Interactive.interactive_clusteredheatmap
