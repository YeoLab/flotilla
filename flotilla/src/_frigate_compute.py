from __future__ import division
__author__ = 'lovci, obot'


"""

metrics, math for study_data analysis


"""

# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from scipy import stats
from _barge_utils import timeout, TimeoutError
from collections import defaultdict
import networkx as nx
import math
import itertools

def switchy_score(array):
    """Transform a 1D array of df scores to a vector of "switchy scores"

    Calculates std deviation and mean of sine- and cosine-transformed
    versions of the array. Better than sorting by just the mean which doesn't
    push the really lowly variant events to the ends.

    Parameters
    ----------
    array : numpy.array
        A 1-D numpy array or something that could be cast as such (like a list)

    Returns
    -------
    float
        The "switchy score" of the study_data which can then be compared to other
        splicing event study_data

    @author Michael T. Lovci
    """
    array = np.array(array)
    variance = 1 - np.std(np.sin(array[~np.isnan(array)] * np.pi))
    mean_value = -np.mean(np.cos(array[~np.isnan(array)] * np.pi))
    return variance * mean_value

def get_switchy_score_order(x):
    """Apply switchy scores to a 2D array of df scores

    Parameters
    ----------
    x : numpy.array
        A 2-D numpy array in the shape [n_events, n_samples]

    Returns
    -------
    numpy.array
        A 1-D array of the ordered indices, in switchy score order
    """
    switchy_scores = np.apply_along_axis(switchy_score, axis=0, arr=x)
    return np.argsort(switchy_scores)


def binify(df, bins):
    """Makes a histogram of each row the provided binsize

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe whose rows you'd like to binify.
    bins : numpy.array
        Bins you would like to use for this data. Must include the final bin
        value, e.g. (0, 0.5, 1) for the two bins (0, 0.5) and (0.5, 1)

    Returns
    -------
    binned : pandas.DataFrame

    Raises
    ------


    """
    ncol = bins.shape[0] - 1
    nrow = df.shape[0]
    binned = np.zeros((nrow, ncol))

    # TODO: make sure this works for numpy matrices
    for i, (name, row) in enumerate(df.iterrows()):
        binned[i, :] = np.histogram(row, bins=bins, normed=True)[0]

    columns = ['{}-{}'.format(i, j) for i, j in zip(bins, bins[1:])]
    binned = pd.DataFrame(binned, index=df.index, columns=columns)
    return binned


def get_regressor(x,y, n_estimators=1500, pCut=0.05, n_tries=5, verbose=False):

    if verbose:
        sys.stderr.write('getting regressor\n')
    clfs = []
    oob_scores = []

    for i in range(n_tries):
        if verbose:
            sys.stderr.write('%d.' % i)

        clf = ExtraTreesRegressor(n_estimators=n_estimators, oob_score=True,
                                  bootstrap=True, max_features='sqrt',
                                  n_jobs=1, random_state=i).fit(x,y)
        clfs.append(clf)
        oob_scores.append(clf.oob_score_)
    clf = clfs[np.argmax(oob_scores)]
    clf.feature_importances = pd.Series(clf.feature_importances_, index=x.columns)

    return clf, oob_scores


def get_boosting_regressor(x,y,verbose=False):
    if verbose:
        sys.stderr.write('getting boosting regressor\n')

    clf = GradientBoostingRegressor(n_estimators=50, subsample=0.6, max_features=100,
                                    verbose=0, learning_rate=0.1, random_state=0).fit(x,y)

    clf.feature_importances = pd.Series(clf.feature_importances_, index=x.columns)
    if verbose:
        sys.stderr.write('finished boosting regressor\n')

    return clf


def get_unstarted_events(mongodb):
    """
    get events that have not been started yet.
    generator sets started to True before returning an event
    """
    go_on = True
    while go_on ==True:

        event = mongodb['list'].find_one({"started":False})

        if event is None:
            go_on=False

        else:
            event['started'] = True
            mongodb['list'].save(event)
            yield event

@timeout(5) #because these sometimes hang
def get_slope(x,y):
    return stats.linregress(x,y)[0]

@timeout(5) #because these sometimes hang
def do_r(s_1, s_2, method=stats.pearsonr, min_items=12):

    """
    do an R calculation, remove items with values missing in either

    input:
    x - predictor vector
    y - target vector

    optional:
    method - method to use (scipy.stats.pearsonr or scipy.stats.spearmanr)

    output:
    r, p (order as determined by the chosen method)

    return (nan, nan) if too few items overlap

    """
    s_1, s_2 = s_1.dropna().align(s_2.dropna(), join='inner')
    if len(s_1) <= min_items:
        return (np.nan, np.nan)
    return method(s_1, s_2)

@timeout(10) #because these sometimes hang
def get_robust_values(x,y):
    """
    get robust linear regression

    input:
    x - predictor vector
    y - target vector

    output:
    intercept, slope, t-statistic, p-value

    """
    import statsmodels.api as sm
    rlm_result = sm.RLM(y, sm.add_constant(x), missing='drop').fit()
    return rlm_result.params[0], rlm_result.params[1], rlm_result.tvalues[0], rlm_result.pvalues[0],

@timeout(5)
def get_dcor(x, y):
    """
    get dcor

    see: https://github.com/andrewdyates/dcor

    input:
    x - predictor vector
    y - target vector

    output:
    dc, dr, dvx, dvy

    """
    import dcor_cpy as dcor
    dc, dr, dvx, dvy = dcor.dcov_all(x,y)
    return dc, dr, dvx, dvy

@timeout(100)
def apply_calc_rs(X, y, method = stats.pearsonr):
    """
    apply R calculation method on each gene separately (for nan values)

    input:
    X - (cells X rpkms) pd.DataFrame
    y - (cells X psi) pd.Series

    output:
    two pd.Series of:
    R coef, p-value
    """

    out_R = pd.Series(index=X.columns, name=y.name)
    out_P = pd.Series(index=X.columns, name=y.name)
    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            r, p = do_r(x, y, method=method)

        except TimeoutError:
            sys.stderr.write("%s r timeout event:%s, gene:%s\n" % (method, y.name, x.name))
            r, p = np.nan, np.nan
        out_R.ix[this_id] = r
        out_P.ix[this_id] = p
    return out_R, out_P

@timeout(220)
def apply_calc_robust(X, y, verbose=False):

    """X and y are dataframes, returns slope, t-value and p-value of robust regression"""

    if verbose:
        sys.stderr.write("getting robust regression\n")
    out_I = pd.Series(index=X.columns, name=y.name) #intercept
    out_S = pd.Series(index=X.columns, name=y.name) #slope
    out_T = pd.Series(index=X.columns, name=y.name) #t-value
    out_P = pd.Series(index=X.columns, name=y.name) #p-value

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            i, s, t, p = get_robust_values(x, y)
        except TimeoutError:
            sys.stderr.write("robust timeout event:%s, gene:%s\n" % (y.name, x.name))
            i, s, t, p = np.nan, np.nan, np.nan, np.nan
        out_I.ix[this_id] = i
        out_S.ix[this_id] = s
        out_T.ix[this_id] = t
        out_P.ix[this_id] = p
    return out_I, out_S, out_T, out_P

@timeout(50)
def apply_calc_slope(X, y, verbose=False):
    """X and y are dataframes, returns slope, t-value and p-value of robust regression"""
    if verbose:
        sys.stderr.write("getting slope\n")

    out_S = pd.Series(index=X.columns, name=y.name)

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            s = get_slope(x, y)
        except TimeoutError:
            sys.stderr.write("linregress timeout event:%s, gene:%s\n" % (y.name, x.name))
            s = np.nan
        out_S.ix[this_id] = s

    return out_S

@timeout(50)
def apply_dcor(X, y, verbose=False):

    if verbose:
        sys.stderr.write("getting dcor\n")

    out_DC = pd.Series(index=X.columns, name=y.name)
    out_DR = pd.Series(index=X.columns, name=y.name)
    out_DVX = pd.Series(index=X.columns, name=y.name)
    out_DVY = pd.Series(index=X.columns, name=y.name)

    for this_id, data in X.iteritems():
        x = pd.Series(data, name=this_id)
        try:
            dc, dr, dvx, dvy = get_dcor(*map(np.array, [x,y]))

        except TimeoutError:
            sys.stderr.write("dcor timeout event:%s, gene:%s\n" % (y.name, x.name))
            dc, dr, dvx, dvy = [np.nan] * 4
        out_DC.ix[this_id] = dc
        out_DR.ix[this_id] = dr
        out_DVX.ix[this_id] = dvx
        out_DVY.ix[this_id] = dvy
    return out_DC, out_DR, out_DVX, out_DVY

# <codecell>

def dropna_mean(x):
    return x.dropna().mean()


import sklearn
from sklearn import decomposition


class Pretty_Reducer(object):
    """

    Just like sklearn's reducers, but with prettied up DataFrames.

    """

    def relabel_pcs(self, x):
        return "pc_" + str(int(x) + 1)

    def fit(self, X):

        try:
            assert type(X) == pd.DataFrame
        except:
            print "Try again as a pandas study_data frame"
            raise

        self.X = X
        super(Pretty_Reducer, self).fit(X)
        self.components_ = pd.DataFrame(self.components_, columns=self.X.columns).rename_axis(self.relabel_pcs, 0)
        try:
            self.explained_variance_ = pd.Series(self.explained_variance_).rename_axis(self.relabel_pcs, 0)
            self.explained_variance_ratio_ = pd.Series(self.explained_variance_ratio_).rename_axis(self.relabel_pcs, 0)
        except AttributeError:
            pass
        return self

    def transform(self, X):
        component_space = super(Pretty_Reducer, self).transform(X)
        if type(self.X) == pd.DataFrame:
                component_space = pd.DataFrame(component_space, index=self.X.index).rename_axis(self.relabel_pcs, 1)
        return component_space

    def fit_transform(self, X):
        try:
            assert type(X) == pd.DataFrame
        except:
            print "Try again as a pandas study_data frame"
            raise
        self.fit(X)
        return self.transform(X)

class PCA(Pretty_Reducer, sklearn.decomposition.PCA):
    pass

class NMF(Pretty_Reducer, sklearn.decomposition.NMF):
    here=True
    def fit(self, X):

        """
        duplicated fit code for NMF because sklearn's NMF cheats for efficiency and calls fit_transform.
        MRO resolves the closest (in this package) fit_transform first and so there's a recursion error:
            def fit(self, X, y=None, **params):

                self.fit_transform(X, **params)
                return self
        """

        try:
            assert type(X) == pd.DataFrame
        except:
            print "Try again as a pandas study_data frame"
            raise

        self.X = X
        super(sklearn.decomposition.NMF, self).fit_transform(X)  #notice this is fit_transform, not fit
        self.components_ = pd.DataFrame(self.components_, columns=self.X.columns).rename_axis(self.relabel_pcs, 0)


class Networker(object):
    weight_funs=['abs', 'sq', 'arctan', 'arctan_sq']

    def get_weight_fun(fun_name):
        _abs =  lambda x: x
        _sq = lambda x: x ** 2
        _arctan = lambda x: np.arctan(x)
        _arctan_sq = lambda x: np.arctan(x) ** 2
        if fun_name == 'abs':
            wt = _abs
        elif fun_name == 'sq':
            wt = _sq
        elif fun_name == 'arctan':
            wt = _arctan
        elif fun_name == 'arctan_sq':
            wt = _arctan_sq
        else:
            raise ValueError
        return wt

    def __init__(self):
        self.adjacencies_ = defaultdict()
        self.graphs_ = defaultdict()
        self._default_node_color_mapper = lambda x: 'r'
        self._default_node_size_mapper = lambda x: 300
        self._last_adjacency_accessed = None
        self._last_graph_accessed = None

    def get_adjacency(self, data=None, name=None, use_pc_1=True, use_pc_2=True,
                      use_pc_3=True, use_pc_4=True, n_pcs=5):

        if data is None and self._last_adjacency_accessed is None:
            raise AttributeError("this hasn't been called yet")

        if name is None:
            if self._last_adjacency_accessed is None:
                name = 'default'
            else:
                name = self._last_adjacency_accessed
        self._last_adjacency_accessed = name
        try:
            if name in self.adjacencies_:
                #print "returning a pre-built adjacency"
                return self.adjacencies_[name]
            else:
                raise ValueError("adjacency hasn't been built yet")
        except ValueError:
            #print 'reduced space', data.shape
            total_pcs = data.shape[1]
            use_cols = np.ones(total_pcs, dtype='bool')
            use_cols[n_pcs:] = False
            use_cols = use_cols * np.array([use_pc_1, use_pc_2, use_pc_3, use_pc_4] + [True,]*(total_pcs-4))
            selected_cols = data.loc[:,use_cols]
            cov = np.cov(selected_cols)
            nRow, nCol = selected_cols.shape
            adjacency = pd.DataFrame(np.tril(cov * - (np.identity(nRow) - 1)),
                                     index=selected_cols.index, columns=data.index)
            #convert to triangular matrix with 0's on diag

            self.adjacencies_[name] = adjacency

        return self.adjacencies_[name]

    def get_graph(self, adjacency=None, cov_cut=None, name=None,
                  node_color_mapper=None,
                  node_size_mapper=None,
                  degree_cut = 2,
                  wt_fun='abs'):

        if node_color_mapper is None:
            node_color_mapper = self._default_node_color_mapper
        if node_size_mapper is None:
            node_size_mapper = self._default_node_size_mapper

        if name is None:
            if self._last_graph_accessed is None:
                name = 'default'
            else:
                name = self._last_graph_accessed
        self._last_graph_accessed = name
        try:
            g,pos = self.graphs_[name]
        except:
            wt = get_weight_fun(wt_fun)
            g = nx.Graph()
            for node_label in adjacency.index:

                node_color = node_color_mapper(node_label)
                node_size = node_size_mapper(node_label)
                g.add_node(node_label, node_size=node_size, node_color=node_color)
        #    g.add_nodes_from(adjacency.index) #to add without setting attributes...neater, but does same thing as above loop
            for cell1, others in adjacency.iterrows():
                for cell2, value in others.iteritems():
                    if value > cov_cut:
                        #cast to floats because write_gml doesn't like numpy dtypes
                        g.add_edge(cell1, cell2, weight=float(wt(value)),inv_weight=float(1/wt(value)), alpha=0.05)

            g.remove_nodes_from([k for k, v in g.degree().iteritems() if v <= degree_cut])

            pos = nx.spring_layout(g)
            self.graphs_[name] = (g, pos)

        return g, pos

class Predictor(object):

    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    extratrees_default_params = {'n_estimators':100,
                               'bootstrap':True,
                               'max_features':'auto',
                               'random_state':0,
                               'verbose':1,
                               'oob_score':True,
                               'n_jobs':2,
                               'verbose':True}

    extratreees_scoring_fun = lambda clf: clf.feature_importances_
    extratreees_scoring_cutoff_fun = lambda scores: np.mean(scores) + 2*np.std(scores) # 2 std's above mean

    from sklearn.ensemble import GradientBoostingClassifier
    boosting_classifier_params = {'n_estimators': 80,  'max_features':1000,  'learning_rate': 0.2,  'subsample': 0.6,}
    boosting_scoring_fun = lambda clf: clf.feature_importances_
    boosting_scoring_cutoff_fun = lambda scores: np.mean(scores) + 2*np.std(scores)

    default_classifier, default_classifier_name = ExtraTreesClassifier, "ExtraTreesClassifier"
    default_regressor, default_regressor_name = ExtraTreesRegressor, "ExtraTreesRegressor"

    default_classifier_scoring_fun = default_regressor_scoring_fun = extratreees_scoring_fun
    default_classifier_scoring_cutoff_fun = default_regressor_scoring_cutoff_fun = extratreees_scoring_cutoff_fun
    default_classifier_params = default_regressor_params = extratrees_default_params

    def __init__(self, data_df, metadata_df,
                 name="Classifier",
                 categorical_traits = None,
                 continuous_traits = None,
                 ):
        """
        train regressors_ or classifiers_ on data.

        name: titles for plots and things...
        sample_list: a list of sample ids for this comparer
        critical_variable: a response variable to test or a list of them
        data_df: pd.DataFrame containing arrays in question
        metadata_df: pd.DataFrame with metadata about data_df
        categorical_traits: which traits are catgorical? - if None, assumed to be all traits
        continuous_traits: which traits are continuous - i.e. build a regressor, not a classifier
        """

        self.has_been_fit_yet=False
        self.has_been_scored_yet=False
        self.name = name
        self.X = data_df
        self.important_features = {}
        self.traits = []
        self.categorical_traits = categorical_traits
        if categorical_traits is not None:
            self.traits.extend(categorical_traits)

        self.continuous_traits = continuous_traits
        if continuous_traits is not None:
            self.traits.extend(continuous_traits)

        print "Initializing predictors for %s" % " and ".join(self.traits)


        #print "Using traits: ", self.traits

        self.trait_data = metadata_df[self.traits] #traits from source, in case they're needed later
        self.X, self.trait_data = self.X.align(self.trait_data, axis=0, join='inner')
        self.y = pd.DataFrame(index=self.X.index, columns=self.traits) #traits encoded to do some work -- "target" variable

        self.classifiers_ = {}
        from sklearn.preprocessing import LabelEncoder

        for trait in self.traits:
            self.important_features[trait] = {}

        for trait in self.categorical_traits:
            try:
                assert len(metadata_df.groupby(trait).describe().index.levels[0]) == 2
            except AssertionError:
                print "WARNING: trait \"%s\" has >2 categories"
            self.classifiers_[trait] = {}
            traitset = metadata_df.groupby(trait).describe().index.levels[0]
            le = LabelEncoder().fit(traitset)  #categorical encoder
            self.y[trait] = le.transform(self.trait_data[trait])  #categorical encoding

        self.continuous_traits = continuous_traits
        self.regressors_ = {}
        if self.continuous_traits is not None:

            for trait in self.continuous_traits:
                self.regressors_[trait] = {}
                self.y[trait] = self.trait_data[trait]

    def fit_classifiers(self,
                        traits=None,
                        classifier_name=default_classifier_name,
                        classifier=default_classifier,
                        classifier_params=default_classifier_params,
                        ):
        """ fit classifiers_ to the data
        traits - list of trait(s) to fit a classifier upon,
        if None, fit all traits that were initialized.
        Classifiers on each trait will be stored in: self.classifiers_[trait]

        classifier_name - a name for this classifier to be stored in self.classifiers_[trait][classifier_name]
        classifier - sklearn classifier object such as ExtraTreesClassifier
        classifier_params - dictionary for paramters to classifier
        """

        if traits is None:
            traits = self.categorical_traits
        else:
            assert type(traits) == list or type(traits) == set
            traits = traits

        for trait in traits:
            clf = classifier(**classifier_params)
            print "Fitting a classifier for trait %s... please wait." %trait
            clf.fit(self.X, self.y[trait])
            self.classifiers_[trait][classifier_name] = clf
            print "Finished..."
        self.has_been_fit_yet=True


    def score_classifiers(self,
                          traits=None,
                          classifier_name=default_classifier_name,
                          feature_scoring_fun=default_classifier_scoring_fun,
                          score_cutoff_fun=default_classifier_scoring_cutoff_fun):
        """
        collect scores from classifiers_
        traits - list of trait(s) to score. Retrieved from self.classifiers_[trait]
        classifier_name - a name for this classifier to be retrieved from self.classifiers_[trait][classifier_name]
        feature_scoring_fun - fxn that yields higher values for better features
        score_cutoff_fun - fxn that that takes output of feature_scoring_fun and returns a cutoff
        """

        if traits is None:
            traits = self.categorical_traits

        for trait in traits:

            try:
                assert trait in self.classifiers_
            except:
                print "trait: %s" % trait, "is missing, continuing"
                continue
            try:
                assert classifier_name in self.classifiers_[trait]
            except:
                print "classifier: %s" % classifier_name, "is missing, continuing"
                continue

            print "Scoring classifier: %s for trait: %s... please wait." % (classifier_name, trait)

            clf = self.classifiers_[trait][classifier_name]
            clf.scores_ = pd.Series(feature_scoring_fun(clf), index=self.X.columns)
            clf.score_cutoff_ = score_cutoff_fun(clf.scores_)
            clf.good_features_ = clf.scores_ > clf.score_cutoff_
            self.important_features[trait][classifier_name] = clf.good_features_
            clf.n_good_features_ = np.sum(clf.good_features_)
            clf.subset_ = self.X.T[clf.good_features_].T

            print "Finished..."
        self.has_been_scored_yet=True

    def fit_regressors(self,
                       traits=None,
                       regressor_name=default_regressor_name,
                       regressor=default_regressor,
                       regressor_params=default_regressor_params,
                      ):
        raise NotImplementedError("Untested, should be close to working.")

        if traits is None:
            traits = self.continuous_traits

        for trait in traits:
            clf = regressor(**regressor_params)
            print "Fitting a classifier for trait %s... please wait." %trait
            clf.fit(self.X, self.y[trait])
            self.regressors_[trait][regressor_name] = clf
            print "Finished..."

    def score_regressors(self,
                          traits=None,
                          regressor_name=default_regressor_name,
                          feature_scoring_fun=default_regressor_scoring_fun,
                          score_cutoff_fun=default_regressor_scoring_cutoff_fun):
        """
        collect scores from classifiers_
        feature_scoring_fun: fxn that yields higher values for better features
        score_cutoff_fun fxn that that takes output of feature_scoring_fun and returns a cutoff
        """
        raise NotImplementedError("Untested, should be close to working.")
        if traits is None:
            traits = self.continuous_traits

        for trait in traits:

            try:
                assert trait in self.regressors_
            except:
                print "trait: %s" % trait, "is missing, continuing"
                continue
            try:
                assert regressor_name in self.regressors_[trait]
            except:
                print "classifier: %s" % regressor_name, "is missing, continuing"
                continue

            print "Scoring classifier: %s for trait: %s... please wait." % (regressor_name, trait)

            clf = self.regressors_[trait][regressor_name]
            clf.scores_ = pd.Series(feature_scoring_fun(clf), index=self.X.columns)
            clf.score_cutoff_ = score_cutoff_fun(clf.scores_)
            self.important_features[trait][regressor_name] = clf.good_features_
            clf.good_features_ = clf.scores_ > clf.score_cutoff_
            clf.n_good_features_ = np.sum(clf.good_features_)
            clf.subset_ = self.X.T[clf.good_features_].T
            print "Finished..."


def benjamini_hochberg(pValues, FDR=0.1):
    """ benjamini-hochberg correction for MHT
        pValues is a list of pValues
        FDR is the desired false-discovery rate

        from: http://udel.edu/~mcdonald/statmultcomp.html
        "One good technique for controlling the false discovery rate was briefly
        mentioned by Simes (1986) and developed in detail by Benjamini and Hochberg (1995).
        Put the individual P-values in order, from smallest to largest. The smallest
        P-value has a rank of i=1, the next has i=2, etc. Then compare each individual
        P-value to (i/m)Q, where m is the total number of tests and Q is the chosen false
        discovery rate. The largest P-value that has P<(i/m)Q is significant,
        and all P-values smaller than it are also significant."

        """
    ranks = np.argsort(np.argsort(pValues))

    nComps = len(pValues) + 0.0
    pSorter = np.argsort(pValues)
    pRank = np.argsort(np.argsort(pValues))+1
    BHcalc = (pRank / nComps) * FDR
    sigs = np.ndarray(shape=(nComps, ), dtype='bool')
    issig = True
    for (p, b, r) in itertools.izip(pValues[pSorter], BHcalc[pSorter], pSorter):
        if p > b:
            issig = False
        sigs[r] = issig
    return sigs


class TwoWayGeneComparisonLocal(object):

    def __init__(self, sample1_name, sample2_name, df, pCut = 0.001,
                 local_fraction = 0.1, bonferroni = True, FDR=None,
                 dtype="RPKM"):
        """ Run a two-sample RPKM experiment.
            Give control sample first, it will go on the x-axis
            df is a pandas dataframe with features (genes) on columns and samples on rows
            sample1 and sample2 are the names of rows in df (sample IDs)
            pCut - P value cutoff
            local_fraction - by default the closest 10% of genes are used for local z-score calculation
            bonferroni - p-values are adjusted for MHT with bonferroni correction
            BH - benjamini-hochberg FDR filtering - check result, proceed with caution. sometimes breaks :(
        """

        sample1 = df.ix[sample1_name]
        sample2 = df.ix[sample2_name]

        sampleNames = (sample1.name, sample2.name)
        self.sampleNames = sampleNames

        sample1 = sample1.replace(0, np.nan).dropna()
        sample2 = sample2.replace(0, np.nan).dropna()

        sample1, sample2 = sample1.align(sample2, join='inner')

        self.sample1 = sample1
        self.sample2 = sample2
        labels = sample1.index

        self.nGenes = len(labels)
        if bonferroni:
            correction = self.nGenes
        else:
            correction = 1

        localCount = int(math.ceil(self.nGenes * local_fraction))
        self.pCut = pCut
        self.upGenes = set()
        self.dnGenes = set()
        self.expressedGenes = set([labels[i] for i, t in enumerate(np.any(np.c_[sample1, sample2] > 1, axis=1)) if t])
        self.log2Ratio = np.log2(sample2 / sample1)
        self.average_expression = (sample2 + sample1)/2.
        self.ranks = np.argsort(np.argsort(self.average_expression))
        self.pValues = pd.Series(index = labels)
        self.localMean = pd.Series(index = labels)
        self.localStd = pd.Series(index = labels)
        self.localZ = pd.Series(index = labels)
        self.dtype=dtype

        for g, r in itertools.izip(self.ranks.index, self.ranks):
            if r < localCount:
                start = 0
                stop = localCount

            elif r > self.nGenes - localCount:
                start = self.nGenes - localCount
                stop = self.nGenes

            else:
                start = r - int(math.floor(localCount/2.))
                stop = r + int(math.ceil(localCount/2.))

            localGenes = self.ranks[self.ranks.between(start, stop)].index
            self.localMean.ix[g] = np.mean(self.log2Ratio.ix[localGenes])
            self.localStd.ix[g] = np.std(self.log2Ratio.ix[localGenes])
            self.pValues.ix[g] = stats.norm.pdf(self.log2Ratio.ix[g],
                                                self.localMean.ix[g],
                                                self.localStd.ix[g]) * correction
            self.localZ.ix[g] = (self.log2Ratio.ix[g]- self.localMean.ix[g])/self.localStd.ix[g]

        data = pd.DataFrame(index = labels)
        data["rank"] = self.ranks
        data["log2Ratio"] = self.log2Ratio
        data["localMean"] = self.localMean
        data["localStd"] = self.localStd
        data["pValue"] = self.pValues

        if FDR == None:
            data["isSig"] = self.pValues < pCut
        else:
            data["isSig"] = benjamini_hochberg(self.pValues, FDR=FDR)

        data["meanExpression"] = self.average_expression
        data["localZ"] = self.localZ
        data[sampleNames[0]] = sample1
        data[sampleNames[1]] = sample2

        self.result_ = data

        for label, (pVal, logratio, isSig) in data.get(["pValue", "log2Ratio", "isSig"]).iterrows():
            if (pVal < pCut) and isSig:
                if logratio > 0:
                    self.upGenes.add(label)
                elif logratio < 0:
                   self.dnGenes.add(label)
                else:
                    raise ValueError

    def gstats(self):
        print "I used a p-value cutoff of %e" %self.pCut
        print "There are", len(self.upGenes), "up-regulated genes in %s vs %s" %(self.sampleNames[1],
                                                                                 self.sampleNames[0])
        print "There are", len(self.dnGenes), "down-regulated genes in %s vs %s" %(self.sampleNames[1],
                                                                                   self.sampleNames[0])
        print "There are", len(self.expressedGenes), "expressed genes in both %s and %s" %self.sampleNames


