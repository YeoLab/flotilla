"""
This file will be auto-imported for every testing session, so you can use
these objects and functions across test files.
"""
from collections import defaultdict
import subprocess

import matplotlib as mpl
import numpy as np
import pytest
import pandas as pd
from scipy import stats
import seaborn as sns









# CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
# SHALEK2013_BASE_URL = 'http://oraw.githubusercontent.com/YeoLab/shalek2013/master'
# # SHALEK2013_BASE_URL = 'http://sauron.ucsd.edu/flotilla_projects/shalek2013'
# CHR22_BASE_URL = 'http://sauron.ucsd.edu/flotilla_projects/neural_diff_chr22'

@pytest.fixture(scope='module')
def RANDOM_STATE():
    """Consistent random state"""
    return 0


@pytest.fixture(scope='module')
def n_samples():
    """Number of samples to create example data from"""
    return 50


@pytest.fixture(scope='module')
def samples(n_samples):
    """Sample ids"""
    return ['sample_{}'.format(i + 1) for i in np.arange(n_samples)]


@pytest.fixture(scope='module', params=[True, False])
def technical_outliers(request, n_samples, samples):
    """If request.param is True, return randomly chosen samples as technical
    outliers, otherwise None"""
    if request.param:
        return np.random.choice(samples,
                            size=np.random.randint(1, int(n_samples / 10.)),
                            replace=False)
    else:
        return None


@pytest.fixture(scope='module', params=[True, False])
def pooled(request, n_samples, samples):
    """If request.param is True, return randomly chosen samples as pooled,
    otherwise None"""
    if request.param:
        return np.random.choice(samples,
                                size=np.random.randint(1,
                                                       int(n_samples / 10.)),
                                replace=False)
    else:
        return None

@pytest.fixture(scope='module', params=[True, False])
def outliers(request, n_samples, samples):
    """If request.param is True, return randomly chosen samples as outliers,
    otherwise None"""
    if request.param:
        return np.random.choice(samples,
                                size=np.random.randint(1,
                                                       int(n_samples / 10.)),
                                replace=False)
    else:
        return None


@pytest.fixture(scope='module', params=[2, 3])
def n_groups(request):
    """Number of phenotype groups.

    For testing that functions work when there's only a few groups
    """
    return request.param


@pytest.fixture(scope='module')
def groups(n_groups):
    """Phenotype group names"""
    return ['group{}'.format(i + 1) for i in np.arange(n_groups)]


@pytest.fixture(scope='module', params=['sorted', 'random'])
def group_order(request, groups):
    """so-called 'logical' order of groups for plotting.

    To test if the user gave a specific order of the phenotypes, e.g.
    by differentiation time
    """
    if request.param == 'sorted':
        return list(sorted(groups))
    else:
        return np.random.permutation(groups)


@pytest.fixture(scope='module')
def colors(n_groups):
    """Colors to use for the samples"""
    return map(mpl.colors.rgb2hex,
               sns.color_palette('husl', n_colors=n_groups))


@pytest.fixture(scope='module')
def group_to_color(group_order, colors):
    """Mapping of groups to colors"""
    return dict(zip(group_order, colors))


@pytest.fixture(scope='module')
def color_ordered(group_order, group_to_color):
    """Colors in the order created by the groups"""
    return [group_to_color[g] for g in group_order]


@pytest.fixture(scope='module', params=['simple', 'different'])
def group_to_marker(request):
    """Mapping of groups to plotting markers"""
    if request.param == 'simple':
        return defaultdict(lambda: 'o')
    else:
        marker_iter = iter(list('ov^<>8sp*hHDd'))
        return defaultdict(lambda: marker_iter.next())


@pytest.fixture(scope='module')
def group_transitions(group_order):
    """List of pairwise transitions between phenotypes, for NMF"""
    return zip(group_order[:-1], group_order[1:])


@pytest.fixture(scope='module', params=['phenotype', 'group'])
def metadata_phenotype_col(request):
    """Which column in the metadata specifies the phenotype"""
    return request.param


@pytest.fixture(scope='module')
def metadata_data(groupby, outliers, pooled, samples,
                  n_samples,
                  metadata_phenotype_col):
    df = pd.DataFrame(index=samples)
    if outliers is not None:
        df['outlier'] = df.index.isin(outliers)
    if pooled is not None:
        df['pooled'] = df.index.isin(pooled)
    df[metadata_phenotype_col] = groupby
    df['subset1'] = np.random.choice([True, False], size=n_samples)
    return df


@pytest.fixture(scope='module')
def metadata_kws(metadata_phenotype_col, group_order, group_to_color,
                 group_to_marker):
    kws = {}
    if metadata_phenotype_col != 'phenotype':
        kws['phenotype_col'] = metadata_phenotype_col
    kws['phenotype_order'] = group_order
    kws['phenotype_to_color'] = group_to_color
    kws['phenotype_to_marker'] = group_to_marker
    return kws


@pytest.fixture(scope='module')
def mapping_stats_number_mapped_col():
    return 'mapped_reads'


@pytest.fixture(scope='module')
def mapping_stats_min_reads_default():
    return 5e5


@pytest.fixture(scope='module', params=[None, 1e6])
def mapping_stats_kws(request, mapping_stats_number_mapped_col):
    kws = {'number_mapped_col': mapping_stats_number_mapped_col}
    if request.param is not None:
        kws['min_reads'] = request.param
    return kws


@pytest.fixture(scope='module')
def mapping_stats_data(samples, technical_outliers,
                       mapping_stats_min_reads_default,
                       mapping_stats_kws,
                       mapping_stats_number_mapped_col):
    min_reads = mapping_stats_kws.get('min_reads',
                                      mapping_stats_min_reads_default)
    df = pd.DataFrame(index=samples)
    df[mapping_stats_number_mapped_col] = 2 * min_reads
    if technical_outliers is not None:
        df.ix[technical_outliers, mapping_stats_number_mapped_col] = \
            .5 * min_reads
    return df


@pytest.fixture(scope='module')
def n_genes():
    return 100


@pytest.fixture(scope='module')
def genes(n_genes):
    return ['gene_{}'.format(i + 1) for i in np.arange(n_genes)]


@pytest.fixture(scope='module')
def n_events():
    return 200


@pytest.fixture(scope='module')
def events(n_events):
    return ['event_{}'.format(i + 1) for i in np.arange(n_events)]


@pytest.fixture(scope='module')
def groupby(groups, samples):
    return dict((sample, np.random.choice(groups)) for sample in samples)


@pytest.fixture(scope='module')
def modality_models():
    parameter = 20.
    rv_included = stats.beta(parameter, 1)
    rv_excluded = stats.beta(1, parameter)
    rv_middle = stats.beta(parameter, parameter)
    rv_uniform = stats.uniform(0, 1)
    rv_bimodal = stats.beta(1. / parameter, 1. / parameter)

    models = {'included': rv_included,
              'excluded': rv_excluded,
              'middle': rv_middle,
              'uniform': rv_uniform,
              'bimodal': rv_bimodal}
    return models


@pytest.fixture(scope='module', params=[0, 1])
def na_thresh(request):
    return request.param


@pytest.fixture(scope='module')
def gene_name():
    return 'gene_name'


@pytest.fixture(scope='module')
def event_name():
    return 'event_name'


@pytest.fixture(scope='module')
def gene_categories():
    return list('ABCDE')


@pytest.fixture(scope='module')
def boolean_gene_categories():
    return list('WXYZ')


# @pytest.fixture(scope='module', params=[False, True])
# def pooled(request):
# return request.param
#
# @pytest.fixture(scope='module', params=[False, True])
# def outlier(request):
# return request.param

@pytest.fixture(scope='module', params=[False, True])
def renamed(request):
    return request.param


@pytest.fixture(scope='module')
def expression_data(samples, genes, groupby, na_thresh):
    df = pd.DataFrame(index=samples, columns=genes)
    df = pd.concat([pd.DataFrame(np.vstack([
        np.random.lognormal(np.random.uniform(0, 5), np.random.uniform(0, 2),
                            df.shape[0]) for _ in df.columns]).T,
                                 index=df.index, columns=df.columns) for
                    name, df in
                    df.groupby(groupby)], axis=0).sort_index()
    if na_thresh > 0:
        df = df.apply(lambda x: x.map(
            lambda i: i if np.random.uniform() >
                           np.random.uniform(0, na_thresh)
            else np.nan), axis=1)
    return df


@pytest.fixture(scope='module')
def expression_data_no_na(samples, genes):
    data = np.random.lognormal(5, 2, size=(len(samples), len(genes)))
    return pd.DataFrame(data, index=samples, columns=genes)


@pytest.fixture(scope='module')
def expression_feature_data(genes, gene_categories,
                            boolean_gene_categories, renamed):
    df = pd.DataFrame(index=genes)
    if renamed:
        df['renamed'] = df.index.map(lambda x: x.replace('gene', 'renamed'))
    df['gene_category'] = df.index.map(lambda x:
                                       np.random.choice(gene_categories))
    for category in boolean_gene_categories:
        p = np.random.uniform()
        df[category] = np.random.choice([True, False], size=df.shape[0],
                                        p=[p, 1 - p])
    return df


@pytest.fixture(scope='module')
def expression_feature_rename_col(renamed):
    if renamed:
        return 'renamed'
    else:
        return None


@pytest.fixture(scope='module', params=[None, 2, 10])
def expression_log_base(request):
    return request.param


@pytest.fixture(scope='module', params=[True, False])
def expression_plus_one(request):
    return request.param


@pytest.fixture(scope='module', params=[-np.inf, 0, 2])
def expression_thresh(request):
    return request.param


@pytest.fixture(scope='module')
def expression_kws(expression_feature_data, expression_feature_rename_col,
                   expression_log_base, expression_plus_one,
                   expression_thresh):
    kws = {}
    kws['feature_data'] = expression_feature_data
    kws['feature_rename_col'] = expression_feature_rename_col
    kws['log_base'] = expression_log_base
    kws['plus_one'] = expression_plus_one
    kws['thresh'] = expression_thresh
    return kws


@pytest.fixture(scope='module')
def true_modalities(events, modality_models, groups):
    data = dict((e, dict((g, (np.random.choice(modality_models.keys())))
                         for g in groups)) for e in events)
    return pd.DataFrame(data)


@pytest.fixture(scope='module')
def splicing_data(samples, events, true_modalities, modality_models,
                  na_thresh, groupby):
    df = pd.DataFrame(index=samples, columns=events)
    df = pd.concat([pd.DataFrame(
        np.vstack([modality_models[modality].rvs(df.shape[0])
                   for modality in true_modalities.ix[group]]).T,
        index=df.index, columns=df.columns)
                    for group, df in df.groupby(groupby)], axis=0)
    if na_thresh > 0:
        df = df.apply(lambda x: x.map(
            lambda i: i if np.random.uniform() >
                           np.random.uniform(0, na_thresh)
            else np.nan), axis=1)
        df = pd.concat([d.apply(
            lambda x: x if np.random.uniform() >
                           np.random.uniform(0, na_thresh / 10)
            else pd.Series(np.nan, index=x.index), axis=1) for group, d in
                        df.groupby(groupby)], axis=0)
    return df.sort_index()


@pytest.fixture(scope='module')
def splicing_data_no_na(samples, events,
                        true_modalities, modality_models, na_thresh, groupby):
    df = pd.DataFrame(index=samples, columns=events)
    df = pd.concat([pd.DataFrame(
        np.vstack([modality_models[modality].rvs(df.shape[0])
                   for modality in true_modalities.ix[group]]).T,
        index=df.index, columns=df.columns)
                    for group, df in df.groupby(groupby)], axis=0)
    return df.sort_index()


@pytest.fixture(scope='module')
def splicing_feature_data(events, genes, gene_name, expression_feature_data,
                          splicing_feature_common_id,
                          renamed):
    df = pd.DataFrame(index=events)
    df[gene_name] = df.index.map(lambda x: np.random.choice(genes))
    df = df.join(expression_feature_data, on=splicing_feature_common_id)
    return df


@pytest.fixture(scope='module')
def splicing_feature_common_id(gene_name):
    return gene_name


@pytest.fixture(scope='module')
def splicing_kws(splicing_feature_data, splicing_feature_common_id,
                 gene_name):
    return {'feature_data': splicing_feature_data,
            'feature_rename_col': gene_name,
            'feature_expression_id_col': splicing_feature_common_id}


@pytest.fixture(scope='module')
def study(request, shalek2013, scrambled_study):
    if request.param == 'shalek2013':
        return shalek2013
    if request.param == 'scrambled_study':
        return scrambled_study


@pytest.fixture(scope='module')
def genelist_path(data_dir):
    return '{}/example_gene_list.txt'.format(data_dir)


@pytest.fixture(scope='module')
def genelist_dropbox_link():
    return 'https://www.dropbox.com/s/652y6hb8zonxe4c/example_gene_list.txt' \
           '?dl=0'


@pytest.fixture(params=['local', 'dropbox'])
def genelist_link(request, genelist_path, genelist_dropbox_link):
    if request.param == 'local':
        return genelist_path
    elif request.param == 'dropbox':
        return genelist_dropbox_link


@pytest.fixture(params=[None, 'gene_category: A',
                        'link',
                        'path'], scope='module')
def feature_subset(request, genelist_dropbox_link, genelist_path):
    from flotilla.util import link_to_list

    name_to_location = {'link': genelist_dropbox_link,
                        'path': genelist_path}

    if request.param is None:
        return request.param
    elif request.param in ('link', 'path'):

        try:
            return link_to_list(name_to_location[request.param])
        except subprocess.CalledProcessError:
            # Downloading the dropbox link failed, aka not connected to the
            # internet, so just test "None" again
            return None
    else:
        # Otherwise, this is a name of a subset
        return request.param


@pytest.fixture(scope='module')
def x_norm():
    """Normally distributed numpy array"""
    n_samples = 20
    n_features = 50
    x = np.random.randn(n_samples * n_features)
    x = x.reshape(n_samples, n_features)
    return x


@pytest.fixture(scope='module')
def df_norm(x_norm):
    """Normally distributed pandas dataframe"""
    nrow, ncol = x_norm.shape
    index = ['sample_{0:02d}'.format(i) for i in range(nrow)]
    columns = ['feature_{0:04d}'.format(i) for i in range(ncol)]
    df = pd.DataFrame(x_norm, index=index, columns=columns)
    return df


@pytest.fixture(scope='module')
def df_nonneg(df_norm):
    """Non-negative data for testing NMF"""
    return df_norm.abs()


@pytest.fixture(scope='module', params=[0, 5])
def metadata_minimum_samples(request):
    return request.param


@pytest.fixture(params=[True, False])
def featurewise(request):
    return request.param


@pytest.fixture(scope='module')
def base_data(expression_data):
    from flotilla.data_model.base import BaseData

    return BaseData(expression_data)


@pytest.fixture(params=[None, 'half', 'all'], scope='module')
def sample_ids(request, base_data):
    if request.param is None:
        return request.param
    elif request.param == 'some':
        half = base_data.data.shape[0] / 2
        return base_data.data.index[:half]
    elif request.param == 'all':
        return base_data.data.index


@pytest.fixture(params=[None, 'half', 'all'], scope='module')
def feature_ids(request, base_data):
    if request.param is None:
        return request.param
    elif request.param == 'some':
        half = base_data.data.shape[1] / 2
        return base_data.data.columns[:half]
    elif request.param == 'all':
        return base_data.data.columns


@pytest.fixture(params=[True, False], scope='module')
def standardize(request):
    return request.param


@pytest.fixture(params=['phenotype: Immature BDMC',
                        'not (phenotype: Immature BDMC)',
                        'pooled'],
                scope='module')
def sample_subset(request):
    return request.param