# from singlesail import parsers
from _ExpressionData import ExpressionData
from _SplicingData import SplicingData
from ...project.project_params import _default_group_id
from ..submarine import Networker_Viz

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

    def pca(self, list_name='default', group_id=_default_group_id):

        """Performs PCA on both expression and splicing data
        """
        raise NotImplementedError
        e_reduced = self.expression.get_reduced(list_name, group_id)
        e_reduced.plot_samples()

        s_reduced = self.splicing.get_reduced(list_name, group_id)
        s_reduced.plot_samples()

    def interactive_pca(self, featurewise=False):
        from IPython.html.widgets import interactive
        from ..submarine import Networker_Viz
        try:
            assert hasattr(self, 'gene_networks')
        except:
            pass
        #    self.gene_networks = super(Networker_Viz, self).__init__()

        #self.gene_networks.draw_graph(self.expression, featurewise=featurewise)


