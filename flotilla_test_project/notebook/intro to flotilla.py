# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline 

# <markdowncell>

# #FLOTILLA INTRODUCTION

# <markdowncell>

# Flotilla is a container for software to do genomics things. It is sub-divided into thematic parts that are named after boats.
# 
# "object-oriented magic" - schooner
# <img src="img/schooner.png" height=200 width=200/>
# 
# "exploratory data viz" - submarine
# <img src="img/submarine.png" height=200 width=200/>
# 
# "things for computation" - frigate
# <img src="img/frigate.png" height=200 width=200/>
# 
# "draw data from external sources" - skiff
# <img src="img/skiff.png" height=200 width=200/>
# 
# "slow-loading things" - cargo
# <img src="img/cargo.png" height=200 width=200/>
# 
# "database things" - carrier
# <img src="img/carrier.png" height=200 width=200/>

# <markdowncell>

# projects are stored in separate git repos, access them by import.
# 
# <p>neural_diff_project is yan's data, but projects can come with readme's<br> that make the stderr show messages about the data (like below)</p>

# <markdowncell>

# ###users import prjoects using:
#  
#  <pre><code>import projectname</code></pre>
#  
#  where someone has already installed projectname into the python path.
#  
#  This tutorial uses neural_diff_project as an example.

# <codecell>

import pandas as pd

# <codecell>

%pdb

# <codecell>

import neural_diff_project

# <codecell>

%pdb

# <codecell>

neural_diff_project.embark()

# <markdowncell>

# 
# 
# STDERR provides messages about the methods used to make the data.

# <markdowncell>

# ##<b>schooner</b> is an object-oriented user interface.<br>
# <b>schooner.Study</b> takes care of organizing data from multiple sources. It inherits subclasses that have tools to interact with data via IPython widgets, which is really nice.<br>
# <b>schooner.ExpressionData</b> and <b>schooner.SplicingData</b> do classifcation/regression and dimensionality reduction.<br>
# All data types should be available for user interaction (and are expected to be used) via the
# <b>schooner.Study</b> object.
# <br>
# <br>
# ##<b>submarine</b> is a visualization module that entirely depends on the other modules,this is mostly subclasses of the modules in frigate<br>
# ##<b>frigate</b> is the computational workhorse, it holds the objects that do the maths.

# <codecell>

sc_study = neural_diff_project.embark(load_cargo=True, drop_outliers=True) 

# <markdowncell>

# ###what just happened?
# this puts the expression and splicing data, along with metadata about the <br> samples into an object-oriented interface called flotilla.schooner.Study which is now accessible through the <code>sc_study</code> variable<br>
# load_cargo=True takes >40 seconds to load the first time, it loads GO references into memory and sets up gene names for plots and such <br>

# <markdowncell>

# #Sample Metadata
# 
# plotting requires that two columns appear in the sample metadata data frame: "color" and "cell_marker"<br>

# <codecell>

sc_study.sample_info

# <markdowncell>

# #Raw data for gene expression, TPM:

# <codecell>

sc_study.expression.expression_df

# <markdowncell>

# #Raw data for splicing, PSI:

# <codecell>

sc_study.splicing.splicing_df

# <markdowncell>

# ##parameters established in <code>neural_cell_diff</code> poulate the lists of interactive objects

# <markdowncell>

# ##<b>interactive_pca</b> 
# 
# runs PCA
# 
# group_id is a subset of samples to use<br>
# data_type -designates whether you'd like to do PCA with either splicing or gene expression as features<br>
# featurewise - selects whether you'd like to visualize PCA of samples or of features<br>
# x_pc and y_pc - select the components to display for the x- and y- dimensions<br>
# show_point_labels - draws sample/feature labels<br>
# list_link - a local file path (full path) or a http: link to a list of line-delimited features to use for the PCA<br>
# list_name - contains pre-loaded lists, you can set this to "custom" to use list_link<br>
# savefile - a file to output
# 

# <codecell>

sc_study.interactive_pca()

# <markdowncell>

# ##<b>interactive_graph</b> 
# 
# makes a graph where layout of nodes is determined by their covariance in PCA-space
# 
# group_id - a subset of samples to use<br>
# data_type - designates whether you'd like to do PCA with either splicing or gene expression as features<br>
# featurewise - selects whether you'd like to visualize samples or of features<br>
# draw_labels - draws sample/feature labels on nodes<br>
# degree_cut - minimum degree for nodes to be included in the output
# cov_std_cut - minimum covariance for two nodes to have an edge
# n_pcs - use the first n_pcs
# feature_of_interest - feature to highlight. recommended that you paste into this box, rather than type directly
# use_pc_X - de-select to exclude component X
# list_name - contains pre-loaded lists and custom lists added with custom "list_link" in interactive_pca<br>
# savefile - a file to output
# weight_fun - a function to apply to covariances before filtering

# <codecell>

sc_study.interactive_graph()

# <markdowncell>

# ##<b>interactive_clf</b> 
# 
# makes a classifier, plots PCA of samples using only important features from the classifier. (default classifier is ExtraTreesClassifier, extremely randomized forests)
# 
# data_type - designates whether you'd like to do PCA with either splicing or gene expression as features<br>
# list_name - contains pre-loaded lists and custom lists added with custom "list_link" in interactive_pca<br>
# group_id - a subset of samples to use<br>
# categorical_variable - a column in expression.sample_descriptors upon which to train a classifier<br>
# feature_score_std_cutoff - mean + (this x std) is used to select the top features<br>
# safefile - a file to output

# <codecell>

sc_study.interactive_clf()

# <codecell>

print "here are the pooled samples\n"
import pandas as pd
for i in sc_study.sample_info.index[pd.Series(sc_study.sample_info['is_pooled'], dtype='bool')]:
    print "\t", i

# <markdowncell>

# ##<b>interatctive_localZ</b>
# 
# runs localZ comparision on 2 RNAseq samples... works on any datatype but only really has meaning on expression.<br>
# sample1 - name of sample on x-axis<br>
# sample2 - name of sample on y-axis<br>
# pCut - p-value cutoff, must be a float. recommended that you paste here, not type in this box.<br>
# 
# running this updates sc_study.localZ_result with a dataFrame of the results to be parsed by the user

# <codecell>

sc_study.interactive_localZ()

# <codecell>

sc_study.localZ_result

# <codecell>


