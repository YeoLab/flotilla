"""
Data models for "studies" studies include attributes about the data and are heavier in terms of data load
"""


from .._submaraine_viz import NetworkerViz, PredictorViz
from .._cargo_commonObjects import Cargo
import matplotlib.pyplot as plt

class Study(Cargo):
    """

    Attributes
    ----------


    Methods
    -------

    """

    _default_x_pc = 1
    _default_y_pc = 2

    def __init__(self, study, load_cargo=True):
        """Constructor for Study object containing gene expression and
        alternative splicing study_data.

        Parameters
        ----------
        sample_info_filename, expression_df, splicing_df

        Returns

        -------


        Raises
        ------

        """
        self.study = study

        self.sample_info = study.sample_info

        self.expression = ExpressionStudy(study, load_cargo=load_cargo)
        self.splicing = SplicingStudy(study)

        self.default_group_id = study.default_group_id
        self.default_group_ids = study.default_group_ids

        self.default_list_id = study.default_list_id
        self.default_list_ids = study.default_list_ids

        self.expression_networks = NetworkerViz(self.expression)
        self.splicing_networks = NetworkerViz(self.splicing)


    def detect_outliers(self):
        """Detects outlier cells from expression, mapping, and splicing study_data and labels the outliers as such for future analysis.

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
        """Performs Jensen-Shannon Divergence on both splicing and expression study_data

        Jensen-Shannon divergence is a method of quantifying the amount of
        change in distribution of one measurement (e.g. a splicing event or a
        gene expression) from one celltype to another.
        """
        self.expression.jsd()
        self.splicing.jsd()


    def pca(self, data_type='expression', x_pc=1, y_pc=2, **kwargs):

        """Performs PCA on both expression and splicing study_data
        """
        if data_type == "expression":
            self.expression.plot_dimensionality_reduction(x_pc=x_pc, y_pc=y_pc, **kwargs)
        elif data_type == "splicing":
            self.splicing.plot_dimensionality_reduction(x_pc=x_pc, y_pc=y_pc, **kwargs)


    def graph(self, data_type='expression', **kwargs):
        args = kwargs.copy()

        if data_type == "expression":
            self.expression_networks.draw_graph(**kwargs)

        elif data_type == "splicing":
            self.splicing_networks.draw_graph(**kwargs)

    def predictor(self, data_type='expression', **kwargs):

        """Performs PCA on both expression and splicing study_data
        """
        if data_type == "expression":
            self.expression.plot_classifier(**kwargs)
        elif data_type == "splicing":
            self.splicing.plot_classifier(**kwargs)


    def interactive_pca(self):
        from IPython.html.widgets import interact

        #not sure why nested fxns are required for this, but they are... i think...
        def do_interact(group_id=self.default_group_id, data_type='expression',
                        featurewise=False, x_pc=1, y_pc=2, show_point_labels=False, list_link = '',
                        list_name=self.default_list_id, savefile = ''):

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

            self.pca(group_id=group_id, data_type=data_type, featurewise=featurewise,
                      x_pc=x_pc, y_pc=y_pc, show_point_labels=show_point_labels, list_name=list_name)
            if savefile != '':
                f = plt.gcf()
                f.savefig(savefile)

        interact(do_interact,
                 data_type=('expression', 'splicing'),
                 group_id=self.default_group_ids,
                 list_name = self.default_list_ids + ["custom"],
                 featurewise=False,
                 x_pc=(1,10),  y_pc=(1,10),
                 show_point_labels=False, )

    def interactive_graph(self):
        from IPython.html.widgets import interact

        #not sure why nested fxns are required for this, but they are... i think...
        def do_interact(group_id=self.default_group_id, data_type='expression',
                        featurewise=False, draw_labels=False, degree_cut=1,
                        cov_std_cut=1.8, n_pcs=5, feature_of_interest="RBFOX2",
                        list_name=self.default_list_id, savefile=''):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                print k, ":", v

            if data_type == 'expression':
                assert(list_name in self.expression.lists.keys())
            if data_type == 'splicing':
                assert(list_name in self.expression.lists.keys())

            self.graph(list_name=list_name, group_id=group_id, data_type=data_type,
                       featurewise=featurewise, draw_labels=draw_labels,
                       degree_cut=degree_cut, cov_std_cut=cov_std_cut, n_pcs = n_pcs,
                       feature_of_interest=feature_of_interest)
            if savefile is not '':
                plt.gcf().savefig(savefile)
        all_lists = list(set(self.expression.lists.keys() + self.splicing.lists.keys()))
        interact(do_interact, group_id=self.default_group_ids,
                data_type=('expression', 'splicing'),
                list_name=all_lists,
                featurewise=False,
                cov_std_cut = (0.1, 3),
                degree_cut = (0,10),
                n_pcs=(2,100),
                draw_labels=False,
                feature_of_interest="RBFOX2"
                )

    def interactive_clf(self):

        from IPython.html.widgets import interact

        #not sure why nested fxns are required for this, but they are... i think...
        def do_interact(data_type='expression',
                        list_name=self.default_list_id, group_id=self.default_group_id,
                        categorical_variable='outlier', feature_score_std_cutoff=2, savefile=''):

            for k, v in locals().iteritems():
                if k == 'self':
                    continue
                print k, ":", v

            if data_type == 'expression':
                data_obj = self.expression
            if data_type == 'splicing':
                data_obj = self.splicing
                print "yay, splicing!"

            assert(list_name in data_obj.lists.keys())

            prd = data_obj.get_predictor(list_name, group_id, categorical_variable)
            prd(categorical_variable, feature_score_std_cutoff=feature_score_std_cutoff)
            print "retrieve this predictor with:\nprd=study.%s.get_predictor('%s', '%s', '%s')\npca=prd('%s', %f)" \
            % (data_type, list_name, group_id, categorical_variable, categorical_variable, feature_score_std_cutoff)
            if savefile is not '':
                plt.gcf().savefig(savefile)
        all_lists = list(set(self.expression.lists.keys() + self.splicing.lists.keys()))
        interact(do_interact,
                data_type=('expression', 'splicing'),
                list_name=all_lists,
                group_id=self.default_group_ids,
                categorical_variable=[i for i in self.default_group_ids if not i.startswith("~")],
                feature_score_std_cutoff = (0.1, 20),
                draw_labels=False,
                feature_of_interest="RBFOX2"
                )
    #TODO:draw_last_graph function.
from _ExpressionData import ExpressionData
from _SplicingData import SplicingData
cargo = Cargo()


class ExpressionStudy(ExpressionData):

    def __init__(self, study, load_cargo=True, **kwargs):

        assert hasattr(study, 'expression')
        assert hasattr(study, 'sample_info')
        assert hasattr(study, 'expression_info')

        super(ExpressionStudy, self).__init__(expression_df=study.expression,
                                             sample_descriptors= study.sample_info,
                                             gene_descriptors=study.expression_info,
                                             **kwargs)
        self.default_group_id = study.default_group_id
        self.default_group_ids = study.default_group_ids
        self.default_list_id = study.default_gene_list
        self.species = study.species
        if load_cargo:
            self.load_cargo()
            self.default_list_id = study.default_gene_list
            study.default_list_ids.extend(self.lists.keys())


    def load_cargo(self, rename=True, **kwargs):
        try:
            species = self.species
            self.cargo = cargo.get_species_cargo(self.species)
            self.go = self.cargo.get_go(species)
            self.lists.update(self.cargo.gene_lists)

            if rename:
                self.set_naming_fun(lambda x: self.go.geneNames(x))
        except:
            raise


class SplicingStudy(SplicingData):

    def __init__(self, study, load_cargo=False, **kwargs):

        assert hasattr(study, 'splicing')
        assert hasattr(study, 'sample_info')
        assert hasattr(study, 'splicing_info')

        super(SplicingStudy, self).__init__(splicing=study.splicing,
                                           sample_descriptors=study.sample_info,
                                           event_descriptors=study.splicing_info,
                                             **kwargs)
        self.default_group_id = study.default_group_id
        self.default_group_ids = study.default_group_ids
        self.default_list_id = study.default_event_list
        self.species = study.species

    def load_cargo(self):
        raise NotImplementedError
