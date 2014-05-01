__author__ = 'lovci'

from collections import defaultdict

import pylab
import seaborn
from sklearn.preprocessing import StandardScaler
import networkx as nx
from matplotlib.gridspec import GridSpec as GridSpec
import numpy as np

from ..project.data import descriptors, rpkms, psi, sparse_rpkms
from ..project.project_params import min_cells
from flotilla.src.frigate import PCA
from flotilla.src.submarine import PCA_viz
from cargo import gene_lists
from skiff import neuro_genes_human
from frigate import dropna_mean
from skiff import link_to_list



#default red nodes, all same size
default_node_color_mapper = lambda x: 'r'
default_node_size_mapper = lambda x: 300

