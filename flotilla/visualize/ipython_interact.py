"""
Named `ipython_interact.py` rather than just `interact.py` to differentiate
between IPython interactive visualizations vs D3 interactive visualizations.
"""
import itertools
import sys
import warnings

from IPython.html.widgets import interact
import matplotlib.pyplot as plt




# from ..compute.predict import default_classifier
from ..visualize.color import red
from .network import NetworkerViz
from .color import str_to_color
from ..util import natural_sort
from ..external import link_to_list

default_classifier = 'ExtraTreesClassifier'
default_regressor = 'ExtraTreesRegressor'
default_score_coefficient = 2


class Interactive(object):
    """

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, *args, **kwargs):
        self._default_x_pc = 1
        self._default_y_pc = 2

    @staticmethod
    def get_feature_subsets(study, data_types):
        """Given a study and list of data types, get the relevant feature
        subsets

        Parameters
        ----------
        study : flotilla.Study
            A study object which

        Returns
        -------


        Raises
        ------

        """
        feature_subsets = ['custom']

        # datatype_to_

        if 'expression' in data_types:
            try:
                feature_subsets.extend(study.expression.feature_subsets.keys())
            except AttributeError:
                pass
        if 'splicing' in data_types:
            try:
                feature_subsets.extend(study.splicing.feature_subsets.keys())
            except AttributeError:
                pass

        # Cast to "set" to get rid of duplicates, then back to list because you
        # can't sort a set, then back to list after sorting because you get
        # an iterator... yeah ....
        feature_subsets = list(natural_sort(list(set(feature_subsets))))

        # Make sure "variant" is first because all datasets have that
        try:
            feature_subsets.pop(feature_subsets.index('variant'))
        except ValueError:
            pass
        feature_subsets.insert(0, 'variant')
        return feature_subsets

    @staticmethod
    def interactive_pca(self, data_types=('expression', 'splicing'),
                        sample_subsets=None,
                        feature_subsets=None,
                        featurewise=False,
                        x_pc=(1, 10), y_pc=(1, 10),
                        show_point_labels=False,
                        list_link='', plot_violins=True,
                        savefile='data/last.pca.pdf'):

        def do_interact(data_type='expression',
                        sample_subset=self.default_sample_subsets,
                        feature_subset=self.default_feature_subset,
                        featurewise=False,
                        list_link='',
                        x_pc=1, y_pc=2,
                        plot_violins=True,
                        show_point_labels=False,
                        savefile='data/last.pca.pdf'):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                sys.stdout.write('{} : {}\n'.format(k, v))

            if feature_subset != "custom" and list_link != "":
                raise ValueError(
                    "Set feature_subset to \"custom\" to use list_link")

            if feature_subset == "custom" and list_link == "":
                raise ValueError("Use a custom list name please")

            if feature_subset == 'custom':
                feature_subset = link_to_list(list_link)

            elif feature_subset not in self.default_feature_subsets[data_type]:
                warnings.warn("This feature_subset ('{}') is not available in "
                              "this data type ('{}'). Falling back on all "

                              "features.".format(feature_subset, data_type))

            pca = self.plot_pca(sample_subset=sample_subset,
                                data_type=data_type,
                                featurewise=featurewise,
                                x_pc=x_pc, y_pc=y_pc,
                                show_point_labels=show_point_labels,
                                feature_subset=feature_subset,
                                plot_violins=plot_violins)
            if savefile != '':
                # Make the directory if it's not already there
                self.maybe_make_directory(savefile)
                # f = plt.gcf()
                pca.reduced_fig.savefig(savefile)

                # add "violins" after the provided filename, but before the
                # extension
                violins_file = "_".join([".".join(savefile.split('.')[:-1]),
                                         'violins']) + "." + \
                                         savefile.split('.')[-1]
                if plot_violins:
                    pca.violins_fig.savefig(violins_file)

            feature_subsets = list(
                set(itertools.chain(*self.default_feature_subsets
                                    .values())))

        # self.plot_study_sample_legend()

        if feature_subsets is None:
            feature_subsets = Interactive.get_feature_subsets(self, data_types)

        if sample_subsets is None:
            sample_subsets = self.default_sample_subsets

        return interact(do_interact,
                 data_type=data_types,
                 sample_subset=sample_subsets,
                 feature_subset=feature_subsets + ['custom'],
                 featurewise=featurewise,
                 x_pc=x_pc, y_pc=y_pc,
                 show_point_labels=show_point_labels,
                 list_link=list_link, plot_violins=plot_violins,
                 savefile=savefile)

    @staticmethod
    def interactive_graph(self, data_types=('expression', 'splicing'),
                          sample_subsets=None,
                          feature_subsets=None,
                          featurewise=False,
                          cov_std_cut=(0.1, 3),
                          degree_cut=(0, 10),
                          n_pcs=(2, 100),
                          draw_labels=False,
                          feature_of_interest="RBFOX2",
                          weight_fun=None,
                          use_pc_1=True, use_pc_2=True, use_pc_3=True,
                          use_pc_4=True,
                          savefile='data/last.graph.pdf'):

        from IPython.html.widgets import interact

        # not sure why nested fxns are required for this, but they are... i
        # think...
        def do_interact(data_type='expression',
                        sample_subset=self.default_sample_subsets,
                        feature_subset=self.default_feature_subsets,
                        weight_fun=NetworkerViz.weight_funs,
                        featurewise=False,
                        use_pc_1=True, use_pc_2=True, use_pc_3=True,
                        use_pc_4=True, degree_cut=1,
                        cov_std_cut=1.8, n_pcs=5,
                        feature_of_interest="RBFOX2",
                        draw_labels=False,
                        savefile='data/last.graph.pdf'):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                sys.stdout.write('{} : {}\n'.format(k, v))

            if data_type == 'expression':
                assert (feature_subset in
                        self.expression.feature_subsets.keys())
            if data_type == 'splicing':
                assert (feature_subset in
                        self.splicing.feature_subsets.keys())

            self.plot_graph(data_type=data_type,
                            sample_subset=sample_subset,
                            feature_subset=feature_subset,
                            featurewise=featurewise, draw_labels=draw_labels,
                            degree_cut=degree_cut, cov_std_cut=cov_std_cut,
                            n_pcs=n_pcs,
                            feature_of_interest=feature_of_interest,
                            use_pc_1=use_pc_1, use_pc_2=use_pc_2,
                            use_pc_3=use_pc_3,
                            use_pc_4=use_pc_4,
                            weight_function=weight_fun)
            if savefile is not '':
                self.maybe_make_directory(savefile)
                plt.gcf().savefig(savefile)

        if feature_subsets is None:
            feature_subsets = Interactive.get_feature_subsets(self, data_types)

        if sample_subsets is None:
            sample_subsets = self.default_sample_subsets
        if weight_fun is None:
            weight_fun = NetworkerViz.weight_funs

        # if not featurewise:
        #     self.plot_study_sample_legend()

        return interact(do_interact,
                 data_type=data_types,
                 sample_subset=sample_subsets,
                 feature_subset=feature_subsets,
                 featurewise=featurewise,
                 cov_std_cut=cov_std_cut,
                 degree_cut=degree_cut,
                 n_pcs=n_pcs,
                 draw_labels=draw_labels,
                 weight_fun=weight_fun,
                 feature_of_interest=feature_of_interest,
                 use_pc_1=use_pc_1, use_pc_2=use_pc_2,
                 use_pc_3=use_pc_3, use_pc_4=use_pc_4,
                 savefile=savefile)

    @staticmethod
    def interactive_classifier(self, data_types=('expression', 'splicing'),
                               sample_subsets=None,
                               feature_subsets=None,
                               categorical_variables=None,
                               predictor_types=None,
                               score_coefficient=(0.1, 20),
                               draw_labels=False,
                               savefile='data/last.clf.pdf'):

        def do_interact(data_type,
                        sample_subset,
                        feature_subset,
                        predictor_type=default_classifier,
                        categorical_variable='outlier',
                        score_coefficient=2,
                        savefile='data/last.clf.pdf'):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                sys.stdout.write('{} : {}\n'.format(k, v))

            self.plot_classifier(trait=categorical_variable,
                                 feature_subset=feature_subset,
                                 sample_subset=sample_subset,
                                 predictor_name=predictor_type,
                                 score_coefficient=score_coefficient,
                                 data_type=data_type)

            if savefile is not '':
                self.maybe_make_directory(savefile)
                plt.gcf().savefig(savefile)

        if feature_subsets is None:
            feature_subsets = Interactive.get_feature_subsets(self, data_types)
            feature_subsets.insert(0, 'variant')
        if sample_subsets is None:
            sample_subsets = self.default_sample_subsets

        if categorical_variables is None:
            categorical_variables = [i for i in self.default_sample_subsets if
                                     not i.startswith(
                                         "~") and i != 'all_samples']

        if predictor_types is None:
            predictor_types = self.predictor_config_manager.predictor_configs.keys()

        # self.plot_study_sample_legend()

        return interact(do_interact,
                 data_type=data_types,
                 sample_subset=sample_subsets,
                 feature_subset=feature_subsets,
                 categorical_variable=categorical_variables,
                 score_coefficient=score_coefficient,
                 draw_labels=draw_labels,
                 predictor_type=predictor_types,
                 savefile=savefile)

    @staticmethod
    def interactive_localZ(self):

        from IPython.html.widgets import interact

        def do_interact(data_type='expression', sample1='', sample2='',
                        pCut='0.01'):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                sys.stdout.write('{} : {}'.format(k, v))

            pCut = float(pCut)
            assert pCut > 0
            if data_type == 'expression':
                data_obj = self.expression
            if data_type == 'splicing':
                data_obj = self.splicing

            try:
                assert sample1 in data_obj.df.index
            except:
                sys.stdout.write("sample: {}, is not in {} DataFrame, "
                                 "try a different sample ID".format(sample1,
                                                                    data_type))
                return
            try:
                assert sample2 in data_obj.df.index
            except:
                sys.stdout.write("sample: {}, is not in {} DataFrame, "
                                 "try a different sample ID".format(sample2,
                                                                    data_type))
                return
            self.localZ_result = data_obj.plot_twoway(sample1, sample2,
                                                      pCut=pCut).result_
            sys.stdout.write("local_z finished, find the result in "
                             "<this_object>.localZ_result_")

        return interact(do_interact,
                 data_type=('expression', 'splicing'),
                 sample1='replaceme',
                 sample2='replaceme',
                 pCut='0.01')

    @staticmethod
    def interactive_plot_modalities_lavalamps(self, sample_subsets=None,
                                              feature_subsets=None,
                                              color=red, x_offset=0,
                                              use_these_modalities=True,
                                              bootstrapped=False,
                                              bootstrapped_kws=None,
                                              savefile=''):
        def do_interact(sample_subset=None, feature_subset=None,
                        color=red, x_offset=0,
                        use_these_modalities=True,
                        bootstrapped=False, bootstrapped_kws=None,
                        savefile=''):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                sys.stdout.write('{} : {}\n'.format(k, v))

            assert (feature_subset in self.splicing.feature_subsets.keys())
            feature_ids = self.splicing.feature_subsets[feature_subset]

            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            n_in_this_class = len(set(
                le.fit_transform(self.experiment_design.data[sample_subset])))

            try:
                assert n_in_this_class
            except:
                raise RuntimeError("this sample designator is not binary")

            sample_series = self.metadata.data[sample_subset]
            # TODO: cast non-boolean binary ids to boolean
            try:
                assert self.experiment_design.data[
                           sample_subset].dtype == 'bool'
            except:
                raise RuntimeError("this sample designator is not boolean")

            sample_ids = self.experiment_design.data[sample_subset].index[
                self.experiment_design.data[sample_subset]]

            self.splicing.plot_modalities_lavalamps(
                sample_ids, feature_ids, color=color, x_offset=x_offset,
                use_these_modalities=use_these_modalities,
                bootstrapped=bootstrapped, bootstrapped_kws=bootstrapped_kws,
                ax=None)
            plt.tight_layout()
            if savefile is not '':
                self.maybe_make_directory(savefile)
                plt.gcf().savefig(savefile)

        if feature_subsets is None:
            feature_subsets = Interactive.get_feature_subsets(self,
                                                              ['splicing'])

        if sample_subsets is None:
            sample_subsets = self.default_sample_subsets

        if bootstrapped_kws is None:
            bootstrapped_kws = {}

        return interact(do_interact,
                 sample_subset=sample_subsets, feature_subset=feature_subsets,
                 color=color, x_offset=x_offset,
                 use_these_modalities=use_these_modalities,
                 bootstrapped=bootstrapped, bootstrapped_kws=bootstrapped_kws,
                 savefile=savefile)

    @staticmethod
    def interactive_lavalamp_pooled_inconsistent(
            self, sample_subsets=None, feature_subsets=None,
            difference_threshold=(0.001, 1.00),
            colors=['red', 'green', 'blue', 'purple', 'yellow'], savefile=''):
        from IPython.html.widgets import interact

        # not sure why nested fxns are required for this, but they are... i
        # think...
        def do_interact(sample_subset=self.default_sample_subsets,
                        feature_subset=self.default_feature_subsets,
                        difference_threshold=0.1,
                        color=red,
                        savefile='data/last.lavalamp_pooled_inconsistent.pdf'):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                sys.stdout.write('{} : {}\n'.format(k, v))

            assert (feature_subset in self.splicing.feature_subsets.keys())
            feature_ids = self.splicing.feature_subsets[feature_subset]
            sample_ids = self.sample_subset_to_sample_ids(sample_subset)

            color = str_to_color[color]

            self.splicing.plot_lavalamp_pooled_inconsistent(
                sample_ids, feature_ids, difference_threshold, color=color)
            plt.tight_layout()
            if savefile is not '':
                self.maybe_make_directory(savefile)
                plt.gcf().savefig(savefile)

        if feature_subsets is None:
            feature_subsets = Interactive.get_feature_subsets(self,
                                                              ['splicing',
                                                               'expression'])

        if sample_subsets is None:
            sample_subsets = self.default_sample_subsets

        return interact(do_interact,
                 sample_subset=sample_subsets,
                 feature_subset=feature_subsets,
                 difference_threshold=difference_threshold,
                 color=colors,
                 savefile='')

    # @staticmethod
    # def interactive_clusteredheatmap(self):
    #     def do_interact(data_type='expression',
    #                     sample_subset=self.default_sample_subsets,
    #                     feature_subset=self.default_feature_subset,
    #                     metric='euclidean',
    #                     linkage_method='median',
    #                     list_link='',
    #                     savefile='data/last.clusteredheatmap.pdf'):
    #
    #         for k, v in locals().iteritems():
    #             if k == 'self':
    #                 continue
    #             sys.stdout.write('{} : {}\n'.format(k, v))
    #
    #         if feature_subset != "custom" and list_link != "":
    #             raise ValueError(
    #                 "set feature_subset to \"custom\" to use list_link")
    #
    #         if feature_subset == "custom" and list_link == "":
    #             raise ValueError("use a custom list name please")
    #
    #         if feature_subset == 'custom':
    #             feature_subset = list_link
    #         elif feature_subset not in self.default_feature_subsets[data_type]:
    #             warnings.warn("This feature_subset ('{}') is not available in "
    #                           "this data type ('{}'). Falling back on all "
    #                           "features.".format(feature_subset, data_type))
    #
    #         self.plot_clusteredheatmap(sample_subset=sample_subset,
    #                                    feature_subset=feature_subset,
    #                                    data_type=data_type,
    #                                    metric=metric,
    #                                    linkage_method=linkage_method)
    #         plt.tight_layout()
    #         if savefile != '':
    #             # Make the directory if it's not already there
    #             self.maybe_make_directory(savefile)
    #             f = plt.gcf()
    #             f.savefig(savefile)
    #
    #     feature_subsets = Interactive.get_feature_subsets(self,
    #                                                       ['expression',
    #                                                        'splicing'])
    #
    #     linkage_method = ('single', 'median', 'centroid')
    #     metric = ('euclidean', 'seuclidean')
    #     interact(do_interact,
    #              data_type=('expression', 'splicing'),
    #              sample_subset=self.default_sample_subsets,
    #              feature_subset=feature_subsets,
    #              metric=metric,
    #              linkage_method=linkage_method)
