# from singlesail import parsers
from _ExpressionData import ExpressionData
from _SplicingData import SplicingData
from ...project.project_params import min_cells, _default_group_id, _default_group_ids, _default_list_id, _default_list_ids

class Study(object):
    """

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, sample_info, expression_df=None,
                 splicing_df=None, mapping_stats_df=None, editing_df=None,
                 event_descriptors=None):
        """Constructor for Study object containing gene expression and
        alternative splicing data.

        Parameters
        ----------
        sample_info_filename, expression_df, splicing_df

        Returns
        -------


        Raises
        ------

        """
        # self._sample_info_filename = sample_info_filename
        # self._expression_matrix_filename = expression_df
        # self._splicing_info_filename = splicing_df
        self.mapping_stats = mapping_stats_df

        self.sample_info = sample_info #parsers.read_sample_info(
        # sample_info_filename)
        self.expression = ExpressionData(expression_df, sample_info)

        self.splicing = SplicingData(splicing_df, sample_info, event_descriptors)
        self._default_group_ids = _default_group_ids
        self._default_group_id = _default_group_id
        self._default_list_ids = _default_list_ids
        self._default_list_id = _default_list_id
        self._default_x_pc = 1
        self._default_y_pc = 1

    def detect_outliers(self):
        """Detects outlier cells from expression, mapping, and splicing data and labels the outliers as such for future analysis.

        Parameters
        ----------
        self

        Returns
        -------


        Raises
        ------

        """
        raise NotImplementedError

    def jsd(self):
        """Performs Jensen-Shannon Divergence on both splicing and expression data

        Jensen-Shannon divergence is a method of quantifying the amount of
        change in distribution of one measurement (e.g. a splicing event or a
        gene expression) from one celltype to another.
        """
        self.expression.jsd()
        self.splicing.jsd()

    def pca(self, data_type='expression', x_pc=1, y_pc=2, **kwargs):

        """Performs PCA on both expression and splicing data
        """
        if data_type == "expression":
            self.expression.plot_dimensionality_reduction(x_pc=x_pc, y_pc=y_pc, **kwargs)
        elif data_type == "splicing":
            self.splicing.plot_dimensionality_reduction(x_pc=x_pc, y_pc=y_pc, **kwargs)

    def graph(self, data_type='expression', **kwargs):
        args = kwargs.copy()

        from ..submarine import Networker_Viz

        if data_type == "expression":
            try:
                assert hasattr(self, 'expression_networks')
            except:
                self.expression_networks = Networker_Viz(self.expression)

            self.expression_networks.draw_graph(**kwargs)

        elif data_type == "splicing":
            try:
                assert hasattr(self, 'splicing_networks')
            except:
                self.splicing_networks = Networker_Viz(self.splicing)

            self.splicing_networks.draw_graph(**kwargs)


    def interactive_pca(self):
        from IPython.html.widgets import interact

        #not sure why nested fxns are required for this, but they are... i think...
        def do_interact(group_id=self._default_group_id, data_type='expression',
                        featurewise=False, x_pc=1, y_pc=2, show_point_labels=False, list_link = '',
                        list_name=self._default_list_id):

            #note that this is not consistent with
            if list_name != "custom" and list_link != "":
                raise ValueError("set list_name to \"custom\" to use list_link")

            if list_link == '':
                list_name = None
            else:
                list_name = list_link


            self.pca(group_id=group_id, data_type=data_type, featurewise=featurewise,
                      x_pc=x_pc, y_pc=y_pc, show_point_labels=show_point_labels, list_name=list_name)

        interact(do_interact,
                 data_type=('expression', 'splicing'),
                 group_id=self._default_group_ids,
                 list_name = self._default_list_ids + ["custom"],
                 featurewise=False,
                 x_pc=(1,10),  y_pc=(1,10),
                 show_point_labels=False, )

    def interactive_graph(self):
        from IPython.html.widgets import interact

        #not sure why nested fxns are required for this, but they are... i think...
        def do_interact(group_id=self._default_group_id, data_type='expression',
                        featurewise=False, draw_labels=False, degree_cut=1,
                        cov_std_cut=1.8, n_pcs=5, feature_of_interest="RBFOX2",
                        list_name=self._default_list_id):

            if data_type == 'expression':
                assert(list_name in self.expression.lists.keys())
            if data_type == 'splicing':
                assert(list_name in self.expression.lists.keys())

            self.graph(list_name=list_name, group_id=group_id, data_type=data_type,
                       featurewise=featurewise, draw_labels=draw_labels,
                       degree_cut=degree_cut, cov_std_cut=cov_std_cut, n_pcs = n_pcs,
                       feature_of_interest=feature_of_interest)

        all_lists = list(set(self.expression.lists.keys() + self.splicing.lists.keys()))
        interact(do_interact, group_id=self._default_group_ids,
                data_type=('expression', 'splicing'),
                list_name=all_lists,
                featurewise=False,
                cov_std_cut = (0.1, 3),
                degree_cut = (0,10),
                n_pcs=(2,100),
                draw_labels=False,
                feature_of_interest="RBFOX2"
                )
