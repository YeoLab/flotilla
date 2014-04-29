__author__ = 'lovci'

from collections import defaultdict

import pylab
import seaborn
from sklearn.preprocessing import StandardScaler
import networkx as nx
from matplotlib.gridspec import GridSpec as GridSpec
import numpy as np

from gscripts.general.analysis_tools import PCA
from flotilla.src.submarine import PCA_viz
from cargo import *
from skiff import neuro_genes_human
from frigate import dropna_mean
from skiff import link_to_list

seaborn.set_context('paper')

gene_lists = dict([('confident_rbps', confident_rbps),
                   ('rbps', rbps),
                   ('splicing_genes', splicing_genes),
                   ('all_genes', pd.Series(map(go.geneNames, rpkms.columns), index = rpkms.columns)),
                   ('marker_genes', pd.Series(map(go.geneNames, neuro_genes_human), index = neuro_genes_human)),
                   ('tfs', tfs)
                  ])

gene_lists['default'] = gene_lists['splicing_genes']

event_lists = dict() # don't have any of these right now.

featurewise_event_pca = defaultdict()

var_cut = 0.2
psi_variant = pd.Index([i for i,j in (psi.var().dropna() > var_cut).iteritems() if j])
event_lists['variant'] = psi_variant
event_lists['default'] = event_lists['variant']

def get_featurewise_event_pca(event_list, letter, psi=psi, min_cells=min_cells):
    try:
        return featurewise_event_pca[event_list][letter]
    except:

        if event_list not in event_lists:
            event_lists[event_list] = link_to_list(event_list)

        event_list = event_lists[event_list]
        sparse_subset = psi.ix[descriptors[letter+"_cell"], event_list]
        frequent = pd.Index([i for i,j in (sparse_subset.count() > min_cells).iteritems() if j])
        sparse_subset = sparse_subset[frequent]
        #fill na with mean for each event
        means = sparse_subset.apply(dropna_mean, axis=0)
        mf_subset = sparse_subset.fillna(means,).fillna(0)
        #whiten, mean-center
        ss = pd.DataFrame(StandardScaler().fit_transform(mf_subset), index=mf_subset.index,
                          columns=mf_subset.columns).rename_axis(go.geneNames, 1)
        #compute pca
        pca_obj = PCA_viz(ss.T, whiten=False)
        pca_obj.means = means

        #add mean gene_expression
        featurewise_event_pca[event_list][letter] = pca_obj
    return featurewise_event_pca[event_list][letter]

cellwise_psi_pca = defaultdict()

def get_cellwise_event_pca(event_list, letter):
    try:
        return cellwise_psi_pca[event_list][letter]
    except:

        if event_list not in event_lists:
            event_lists[event_list] = link_to_list(event_list)
        other = event_lists[event_list]

        variant = pd.Index([i for i,j in (psi.var().dropna() > var_cut).iteritems() if j])
        event_list = variant
        sparse_subset = psi.ix[descriptors[letter+"_cell"], event_list]
        frequent = pd.Index([i for i,j in (sparse_subset.count() > min_cells).iteritems() if j])
        sparse_subset = sparse_subset[frequent]
        #fill na with mean for each gene
        means = sparse_subset.apply(dropna_mean, axis=0)
        mf_subset = sparse_subset.fillna(means,).fillna(0)
        #whiten, mean-center
        ss = pd.DataFrame(StandardScaler().fit_transform(mf_subset), index=mf_subset.index,
                          columns=mf_subset.columns).rename_axis(go.geneNames, 1)
        #compute pca
        pca_obj = PCA_viz(ss, whiten=False)
        pca_obj.means = means

        #add mean gene_expression
        cellwise_psi_pca[event_list][letter] = pca_obj
    return cellwise_psi_pca[event_list][letter]

featurewise_gene_pca = defaultdict()

def get_featurewise_gene_pca(gene_list, letter):
    try:
        return featurewise_gene_pca[gene_list][letter]
    except:
        print gene_list

        if gene_list not in gene_lists:
            gene_lists[gene_list] = link_to_list(gene_list)
        other = gene_lists[gene_list]

        sparse_subset = sparse_rpkms.ix[descriptors[letter+"_cell"], pd.Series(other.index).dropna().values]
        #fill na with mean for each gene
        means = sparse_subset.apply(dropna_mean, axis=0)
        mf_subset = sparse_subset.fillna(means,).fillna(0)
        #whiten, mean-center
        ss = pd.DataFrame(StandardScaler().fit_transform(mf_subset), index=mf_subset.index,
                          columns=mf_subset.columns).rename_axis(go.geneNames, 1)
        #compute pca
        pca_obj = PCA_viz(ss.T, whiten=False)
        pca_obj.means = means.rename_axis(go.geneNames)

        #add mean gene_expression
        featurewise_gene_pca[gene_list][letter] = pca_obj
    return featurewise_gene_pca[gene_list][letter]

cellwise_gene_pca = defaultdict(dict)

def get_cellwise_gene_pca(other_name, letter):
    try:
        return cellwise_gene_pca[other_name][letter]
    except:

        if other_name not in gene_lists:
            gene_lists[other_name] = link_to_list(other_name)
        other = gene_lists[other_name]

        sparse_subset = sparse_rpkms.ix[descriptors[letter+"_cell"], pd.Series(other.index).dropna().values]
        #fill na with mean for each gene
        mf_subset = sparse_subset.fillna( sparse_subset.apply(dropna_mean, axis=0),).fillna(0)
        #whiten, mean-center
        ss = pd.DataFrame(StandardScaler().fit_transform(mf_subset), index=mf_subset.index,
                          columns=mf_subset.columns).rename_axis(go.geneNames, 1)
        #compute pca

        cellwise_gene_pca[other_name][letter] = PCA_viz(ss, whiten=False)
    return cellwise_gene_pca[other_name][letter]

adjacencies = defaultdict()
def get_adjacency(pca_space, name=None, pc_1=True, pc_2=True, pc_3=True, pc_4=True,
                  n_pcs=5,):

    try:
        return adjacencies[name]
    except:
        total_pcs = pca_space.shape[1]
        use_pc = np.ones(total_pcs, dtype='bool')
        use_pc[n_pcs:] = False
        use_pc = use_pc * np.array([pc_1, pc_2, pc_3, pc_4] + [True,]*(total_pcs-4))

        good_pc_space = pca_space.loc[:,use_pc]
        cell_rbpPCA_cov = np.cov(good_pc_space)
        nRow, nCol = good_pc_space.shape
        adjacency = pd.DataFrame(np.tril(cell_rbpPCA_cov * -(np.identity(nRow) - 1)),
                                 index=good_pc_space.index, columns=pca_space.index)
        adjacencies[name] = adjacency
    return adjacencies[name]


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

#default red nodes, all same size
default_node_color_mapper = lambda x: 'r'
default_node_size_mapper = lambda x: 300

graphs = defaultdict()

def get_graph(adjacency, cov_cut, graph_name,
              node_color_mapper=default_node_color_mapper,
              node_size_mapper=default_node_size_mapper,
              degree_cut = 2,
              wt_fun='abs'):


    try:
        g,pos = graphs[graph_name]
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
        graphs[graph_name] = (g, pos)

    return g, pos


def dict_to_str(dic):
    """join dictionary data into a string with that data"""
    return "_".join([k+ ":" + str(v) for (k,v) in dic.items()])


def draw_graph(list_name='default', cell_type='any',
               x_pc=1, y_pc=2,
               n_pcs=5,
               pc_1=True, pc_2=True, pc_3=True, pc_4=True,
               degree_cut=2, cov_std_cut = 1.8,
               wt_fun = 'abs',
               cell_components=True, #else feature_components
               gene_features=True, #else event features

               feature_of_interest='RBFOX2', custom_list='', draw_labels=True,
               graph_file=''):

    """
    genelist_name - name of genelist used in making pcas
    cell_type - celltype code
    x_pc - x component for PCA
    y_pc - y component for PCA
    n_pcs - n components to use for cells' covariance calculation
    cov_std_cut - covariance cutoff for edges
    pc{1-4} use these pcs in cov calculation (default True)
    degree_cut - miniumum degree for a node to be included in graph display
    wt_fun - weight function (arctan (arctan cov), sq (sq cov), abs (abs cov), arctan_sq (sqared arctan of cov))
    gene_of_interest - map a gradient representing this gene's rpkm onto nodes



    """
    node_color_mapper = default_node_color_mapper
    node_size_mapper = default_node_size_mapper
    settings = locals().copy()
    #not pertinent to the graph, these are what we want to be able to re-apply to the same graph if it exists
    pca_settings = dict()
    pca_settings['letter'] = cell_type

    adjacency_settings = dict((k, settings[k]) for k in ['pc_1', 'pc_2', 'pc_3', 'pc_4', 'n_pcs'])

    #del settings['gene_of_interest']
    #del settings['graph_file']
    #del settings['draw_labels']

    f = pylab.figure(figsize=(24,18))
    gs = GridSpec(3, 4)
    ax1 = pylab.subplot(gs[0,0:2])
    ax2 = pylab.subplot(gs[0,3])
    ax3 = pylab.subplot(gs[0,2])
    ax4 = pylab.subplot(gs[1:,:2])
    ax5 = pylab.subplot(gs[1:,2:])

   # fname = ":".join([k + "-" + str(v) for (k,v) in locals().iteritems()])

    if custom_list != '':
        list_name = custom_list

    #decide which type of analysis to do.
    if cell_components:
        if gene_features:
            get_pca = get_cellwise_gene_pca
        else:
            get_pca = get_cellwise_event_pca
    else:
        if gene_features:
            #default red nodes, all same size
            get_pca = get_featurewise_gene_pca
        else:
            get_pca = get_featurewise_event_pca

    pca = get_pca(list_name, cell_type)

    pca(show_point_labels=False,
        markers_size_dict=lambda x: 400,
        title=cell_type + " cells", ax=ax1, show_vectors=False,
        title_size=10,
        axis_label_size=10,
        x_pc = "pc_" + x_pc,#this only affects the plot, not the data.
        y_pc = "pc_" + y_pc,#this only affects the plot, not the data.
        )

    if cell_components:
        node_color_mapper = lambda x: descriptors.cell_color[x]
        node_size_mapper = lambda x: 300
        if gene_features:
            pass
        else:
            pass
    else:
        node_color_mapper = lambda x: 'r' if x == feature_of_interest else 'k'
        node_size_mapper = lambda x: pca.X.feature_of_interest.ix[x]
        if gene_features:
            pass
        else:
            pass

    ax3.plot(pca.explained_variance_ratio_ * 100.)
    ax3.axvline(n_pcs)
    ax3.set_ylabel("% explained variance")
    ax3.set_xlabel("component")
    adjacency_name = "_".join(map(dict_to_str, [adjacency_settings, pca_settings]))
    #adjacency_settings['name'] = adjacency_name
    adjacency = get_adjacency(pca.pca_space, name=adjacency_name, **adjacency_settings)
    #f.savefig("tmp/" + fname + ".pca.png")
    cov_dist = np.array([i for i in adjacency.values.ravel() if np.abs(i) > 0])
    cov_cut = np.mean(cov_dist) + cov_std_cut * np.std(cov_dist)

    graph_settings = dict((k, settings[k]) for k in ['wt_fun', 'degree_cut', ])
    graph_settings['cov_cut'] = cov_cut

    this_graph_name = "_".join(map(dict_to_str, [pca_settings, adjacency_settings, graph_settings]))

    seaborn.kdeplot(cov_dist, ax=ax2)
    ax2.axvline(cov_cut)

    g, pos = get_graph(adjacency, cov_cut, this_graph_name, **graph_settings)

    nx.draw_networkx_nodes(g, pos, node_color=map(node_color_mapper, g.nodes()),
                           node_size=map(node_size_mapper, g.nodes()),
                           ax=ax4, alpha=0.5)
    nx.draw_networkx_nodes(g, pos, node_color=map(node_color_mapper, g.nodes()),
                           node_size=map(node_size_mapper, g.nodes()),
                           ax=ax5, alpha=0.5)
    try:
        nx.draw_networkx_nodes(g, pos, node_color=map(lambda x: pca.X[feature_of_interest].ix[x], g.nodes()),
                           cmap=pylab.cm.Greys,
                           node_size=map(lambda x: node_size_mapper(x) * .75, g.nodes()), ax=ax5, alpha=1)
    except:
        pass
    nmr = lambda x:x
    labels = dict([(nm, nmr(nm)) for nm in g.nodes()])
    if draw_labels:
        nx.draw_networkx_labels(g, pos, labels = labels, ax=ax4)
    #mst = nx.minimum_spanning_tree(g, weight='inv_weight')
    nx.draw_networkx_edges(g, pos,ax = ax4,alpha=0.1)
    #nx.draw_networkx_edges(g, pos, edgelist=mst.edges(), edge_color="m", edge_width=200, ax=ax4)
    ax4.set_axis_off()
    ax5.set_axis_off()
    f.tight_layout(pad=5)
    if graph_file != '':
        try:
            nx.write_gml(g, graph_file)
        except Exception as e:
            print "error writing graph file:"
            print e

    return g#, mst