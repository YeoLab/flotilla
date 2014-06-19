"""
Named `ipython_interact.py` rather than just `interact.py` to differentiate
between IPython interactive visualizations vs D3 interactive visualizations.
"""

import matplotlib.pyplot as plt
from .network import NetworkerViz
from .decomposition import DecompositionViz


class Interactive(object):
    """

    Attributes
    ----------


    Methods
    -------

    """
    def __init__(self, *args, **kwargs):
        # super(InteractiveStudy, self).__init__(*args, **kwargs)
        self._default_x_pc = 1
        self._default_y_pc = 2
        [self.minimal_study_parameters.add(param) for param in  ['default_group_id', 'default_group_ids',
                                                                    'default_list_id', 'default_list_ids',]]
        [self.minimal_study_parameters.add(i) for i in ['phenotype_data', ]]
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
                        list_name=self.default_list_id,
                        weight_fun=NetworkerViz.weight_funs,
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
            self.localZ_result = data_obj.plot_twoway(sample1, sample2,
                                                 pCut=pCut).result_
            print "local_z finished, find the result in <this_obj>.localZ_result_"

        interact(do_interact,
                 data_type=('expression', 'splicing'),
                 sample1='replaceme',
                 sample2='replaceme',
                 pCut='0.01')
