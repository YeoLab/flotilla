"""
Named `ipython_interact.py` rather than just `interact.py` to differentiate
between IPython interactive visualizations vs D3 interactive visualizations.
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import os
# import sys
import warnings

import ipywidgets
import matplotlib.pyplot as plt

from .network import NetworkerViz
from ..util import natural_sort, link_to_list


default_classifier = 'ExtraTreesClassifier'
default_regressor = 'ExtraTreesRegressor'


def _print_locals(locals_iteritems):
    print("locals:")
    for k, v in locals_iteritems:
        if k == 'self':
            continue
        print(k, ":", v)


def get_feature_subsets_options(study, data_types):
    """Given a study and list of data types, get the relevant feature
    subsets
    """
    feature_subsets = ['custom']
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
    # first remove 'variant' if it is there, then add it at the front
    try:
        feature_subsets.pop(feature_subsets.index('variant'))
    except ValueError:
        pass
    feature_subsets.insert(0, 'variant')
    return feature_subsets


def interactive_pca(study,
                    data_types=('expression', 'splicing'),
                    sample_subsets=None,
                    feature_subsets=None,
                    color_samples_by=None,
                    featurewise=False,
                    x_pc=(1, 10),
                    y_pc=(1, 10),
                    show_point_labels=False,
                    list_link='', plot_violins=False,
                    scale_by_variance=True,
                    savefile='figures/last.pca.pdf'):
    """
    interactive_pca
    """

    # study.plot_study_sample_legend()
    if feature_subsets is None:
        feature_subsets = get_feature_subsets_options(study, data_types)
    if sample_subsets is None:
        sample_subsets = study.default_sample_subsets
    color_samples_by = study.metadata.data.columns.tolist()

    def do_interact(data_type='expression',
                    sample_subset=study.default_sample_subsets,
                    feature_subset=study.default_feature_subsets,
                    featurewise=False,
                    list_link='',
                    x_pc=1,
                    y_pc=2,
                    plot_violins=False,
                    show_point_labels=False,
                    color_samples_by=study.metadata.phenotype_col,
                    bokeh=False,
                    scale_by_variance=True,
                    most_variant_features=False,
                    std_multiplier=(0, 5.0)):
        _print_locals(locals().iteritems())
        if feature_subset != "custom" and list_link != "":
            raise ValueError(
                "Set feature_subset to \"custom\" if you use list_link")
        if feature_subset == "custom" and list_link == "":
            raise ValueError("Use a list_link if feature_subset is \"custom\"")
        if feature_subset == 'custom':
            feature_subset = link_to_list(list_link)
        elif feature_subset not in study.default_feature_subsets[data_type]:
            warnings.warn("This feature_subset ('{}') is not available in "
                          "this data type ('{}'). Falling back on all "
                          "features.".format(feature_subset, data_type))
        return study.plot_pca(sample_subset=sample_subset,
                              data_type=data_type,
                              featurewise=featurewise,
                              x_pc=x_pc,
                              y_pc=y_pc,
                              show_point_labels=show_point_labels,
                              feature_subset=feature_subset,
                              plot_violins=plot_violins,
                              color_samples_by=color_samples_by,
                              bokeh=bokeh,
                              std_multiplier=std_multiplier,
                              scale_by_variance=scale_by_variance,
                              most_variant_features=most_variant_features)

    gui = ipywidgets.interact(do_interact,
                              data_type=data_types,
                              sample_subset=sample_subsets,
                              feature_subset=feature_subsets + ['custom'],
                              featurewise=featurewise,
                              x_pc=x_pc,
                              y_pc=y_pc,
                              show_point_labels=show_point_labels,
                              list_link=list_link, plot_violins=plot_violins,
                              color_samples_by=color_samples_by,
                              scale_by_variance=scale_by_variance)

    def save(w):
        # Make the directory if it's not already there
        filename, extension = os.path.splitext(savefilename_widget.value)
        extension = extension[1:]
        study.maybe_make_directory(savefilename_widget.value)
        gui.widget.result.fig_reduced.savefig(savefilename_widget.value,
                                              format=extension)
        # add "violins" after provided filename, before extension
        violins_file = '{}.{}'.format("_".join([filename, 'violins']),
                                      extension)
        try:
            gui.widget.result.fig_violins.savefig(
                violins_file, format=extension.lstrip('.'))
        except AttributeError:
            pass

    html_widget = ipywidgets.HTML(value="<hr>")
    savefilename_widget = ipywidgets.Text(description="file name",
                                          value="last_pca.pdf")
    savebutton_widget = ipywidgets.Button(description="save figure below")
    gui.widget.children = list(gui.widget.children) + [html_widget,
                                                       savefilename_widget,
                                                       savebutton_widget]
    savebutton_widget.on_click(save)
    return gui


def interactive_graph(study, data_types=('expression', 'splicing'),
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
                      use_pc_4=True):

    # TODO not sure why nested functions are required for this
    def do_interact(data_type='expression',
                    sample_subset=study.default_sample_subsets,
                    feature_subset=study.default_feature_subsets,
                    weight_fun=NetworkerViz.weight_funs,
                    featurewise=False,
                    use_pc_1=True, use_pc_2=True, use_pc_3=True,
                    use_pc_4=True, degree_cut=1,
                    cov_std_cut=1.8, n_pcs=5,
                    feature_of_interest="RBFOX2",
                    draw_labels=False):
        _print_locals(locals().iteritems())
        if data_type == 'expression':
            assert (feature_subset in
                    study.expression.feature_subsets.keys())
        if data_type == 'splicing':
            assert (feature_subset in
                    study.splicing.feature_subsets.keys())
        study.plot_graph(data_type=data_type,
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

    if feature_subsets is None:
        feature_subsets = get_feature_subsets_options(study, data_types)
    if sample_subsets is None:
        sample_subsets = study.default_sample_subsets
    if weight_fun is None:
        weight_fun = NetworkerViz.weight_funs
    gui = ipywidgets.interact(do_interact,
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
                              use_pc_3=use_pc_3, use_pc_4=use_pc_4)

    def save(w):
        # Make the directory if it's not already there
        filename, extension = os.path.splitext(savefilename_widget.value)
        extension = extension[1:]
        study.maybe_make_directory(savefilename_widget.value)
        plt.gcf().savefig(savefilename_widget.value,
                          format=extension.lstrip('.'))

    html_widget = ipywidgets.HTML(value="<hr>")
    savefilename_widget = ipywidgets.Text(description='file name',
                                          value="last_graph.pdf")
    savebutton_widget = ipywidgets.Button(description="save figure below")
    savebutton_widget.on_click(save)
    gui.widget.children = list(gui.widget.children) + [html_widget,
                                                       savefilename_widget,
                                                       savebutton_widget]
    return gui


def interactive_classifier(study, data_types=('expression', 'splicing'),
                           sample_subsets=None,
                           feature_subsets=None,
                           categorical_variables=None,
                           predictor_types=None,
                           score_coefficient=(0.1, 20),
                           draw_labels=False):

    def do_interact(data_type,
                    sample_subset,
                    feature_subset,
                    predictor_type=default_classifier,
                    categorical_variable='outlier',
                    score_coefficient=2,
                    plot_violins=False,
                    show_point_labels=False):
        _print_locals(locals().iteritems())
        study.plot_classifier(trait=categorical_variable,
                              feature_subset=feature_subset,
                              sample_subset=sample_subset,
                              predictor_name=predictor_type,
                              score_coefficient=score_coefficient,
                              data_type=data_type,
                              plot_violins=plot_violins,
                              show_point_labels=show_point_labels)

    if feature_subsets is None:
        feature_subsets = get_feature_subsets_options(study, data_types)
        feature_subsets.insert(0, 'variant')
    if sample_subsets is None:
        sample_subsets = study.default_sample_subsets
    if categorical_variables is None:
        categorical_variables = [i for i in study.default_sample_subsets
                                 if
                                 not i.startswith("~") and i != 'all_samples']
    if predictor_types is None:
        predictor_types = \
            study.predictor_config_manager.predictor_configs.keys()
    # study.plot_study_sample_legend()
    gui = ipywidgets.interact(do_interact,
                              data_type=data_types,
                              sample_subset=sample_subsets,
                              feature_subset=feature_subsets,
                              categorical_variable=categorical_variables,
                              score_coefficient=score_coefficient,
                              draw_labels=draw_labels,
                              predictor_type=predictor_types)

    def save(w):
        # Make the directory if it's not already there
        filename, extension = os.path.splitext(savefilename_widget.value)
        extension = extension[1:]
        study.maybe_make_directory(savefilename_widget.value)
        gui.widget.result.fig_reduced.savefig(savefilename_widget.value,
                                              format=extension)
        # add "violins" after provided filename, before extension
        violins_file = '{}.{}'.format("_".join([filename, 'violins']),
                                      extension)
        try:
            gui.widget.result.fig_violins.savefig(
                violins_file, format=extension.lstrip('.'))
        except AttributeError:
            pass

    html_widget = ipywidgets.HTML(value="<hr>")
    savefilename_widget = ipywidgets.Text(description='file name',
                                          value="last_classifier.pdf")
    savebutton_widget = ipywidgets.Button(description="save figure below")
    savebutton_widget.on_click(save)
    gui.widget.children = list(gui.widget.children) + [html_widget,
                                                       savefilename_widget,
                                                       savebutton_widget]
    return gui


def interactive_clustermap(study):

    def do_interact(data_type='expression',
                    sample_subset=study.default_sample_subsets,
                    feature_subset=study.default_feature_subset,
                    metric='euclidean',
                    method='median',
                    list_link='',
                    scale_fig_by_data=True,
                    fig_width='', fig_height=''):
        _print_locals(locals().iteritems())
        if feature_subset != "custom" and list_link != "":
            raise ValueError(
                "set feature_subset to \"custom\" to use list_link")
        if feature_subset == "custom" and list_link == "":
            raise ValueError("use a custom list name please")
        if feature_subset == 'custom':
            feature_subset = list_link
        elif feature_subset not in study.default_feature_subsets[data_type]:
            warnings.warn("This feature_subset ('{}') is not available in "
                          "this data type ('{}'). Falling back on all "
                          "features.".format(feature_subset, data_type))
        return study.plot_clustermap(
            sample_subset=sample_subset, feature_subset=feature_subset,
            data_type=data_type, metric=metric, method=method,
            scale_fig_by_data=scale_fig_by_data)

    feature_subsets = get_feature_subsets_options(study,
                                                  ['expression', 'splicing'])
    method = ('average', 'weighted', 'single', 'complete', 'ward')
    metric = ('euclidean', 'seuclidean', 'sqeuclidean', 'chebyshev',
              'cosine', 'cityblock', 'mahalonobis', 'minowski', 'jaccard')
    gui = ipywidgets.interact(do_interact,
                              data_type=('expression', 'splicing'),
                              sample_subset=study.default_sample_subsets,
                              feature_subset=feature_subsets,
                              metric=metric,
                              method=method)

    def save(w):
        filename, extension = os.path.splitext(savefilename_widget.value)
        extension = extension[1:]
        study.maybe_make_directory(savefilename_widget.value)
        gui.widget.result.savefig(savefilename_widget.value,
                                  format=extension.lstrip('.'))

    html_widget = ipywidgets.HTML(value="<hr>")
    savefilename_widget = ipywidgets.Text(description='file name',
                                          value="last_clustermap.pdf")
    savebutton_widget = ipywidgets.Button(description="save figure below")
    savebutton_widget.on_click(save)
    gui.widget.children = list(gui.widget.children) + [html_widget,
                                                       savefilename_widget,
                                                       savebutton_widget]
    return gui


def interactive_correlations(study):

    def do_interact(data_type='expression',
                    sample_subset=study.default_sample_subsets,
                    feature_subset=study.default_feature_subset,
                    metric='euclidean', method='average',
                    list_link='',
                    scale_fig_by_data=True,
                    fig_width='', fig_height='', featurewise=False):
        _print_locals(locals().iteritems())
        if feature_subset != "custom" and list_link != "":
            raise ValueError(
                "set feature_subset to \"custom\" to use list_link")
        if feature_subset == "custom" and list_link == "":
            raise ValueError("use a custom list name please")
        if feature_subset == 'custom':
            feature_subset = list_link
        elif feature_subset not in study.default_feature_subsets[data_type]:
            warnings.warn("This feature_subset ('{}') is not available in "
                          "this data type ('{}'). Falling back on all "
                          "features.".format(feature_subset, data_type))
        return study.plot_correlations(
            sample_subset=sample_subset, feature_subset=feature_subset,
            data_type=data_type, scale_fig_by_data=scale_fig_by_data,
            method=method, metric=metric, featurewise=featurewise)

    feature_subsets = get_feature_subsets_options(study,
                                                  ['expression', 'splicing'])
    method = ('average', 'weighted', 'single', 'complete', 'ward')
    metric = ('euclidean', 'seuclidean', 'sqeuclidean', 'chebyshev',
              'cosine', 'cityblock', 'mahalonobis', 'minowski', 'jaccard')
    gui = ipywidgets.interact(do_interact,
                              data_type=('expression', 'splicing'),
                              sample_subset=study.default_sample_subsets,
                              feature_subset=feature_subsets,
                              metric=metric,
                              method=method,
                              featurewise=False)

    def save(w):
        filename, extension = os.path.splitext(savefilename_widget.value)
        extension = extension[1:]
        study.maybe_make_directory(savefilename_widget.value)
        gui.widget.result.savefig(savefilename_widget.value,
                                  format=extension.lstrip('.'))

    html_widget = ipywidgets.HTML(value="<hr>")
    savefilename_widget = ipywidgets.Text(description='file name',
                                          value="last_correlations.pdf")
    savebutton_widget = ipywidgets.Button(description="save figure below")
    savebutton_widget.on_click(save)
    gui.widget.children = list(gui.widget.children) + [html_widget,
                                                       savefilename_widget,
                                                       savebutton_widget]
    return gui
