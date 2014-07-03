# Roadmap: Flotilla

(Format copied from https://github.com/ipython/ipython/wiki/Roadmap:-IPython)

This document describes the goals, vision and direction of the Flotilla
project. Flotilla is a biological data analysis toolkit aiming to ease
integration of various datatypes and datasets together to solve biological
questions, and put the biological interpretation of computational results
into the hands of experimental biologists, and leave the algorithms and
development to the computational biologists. Additionally,
we see the use of the IPython notebook with its plotting and interactivity as
 a "gateway drug" for biologists to learn to code and invent their own
 analyses. Currently, Flotilla is focused on **single-cell RNA-Seq** analyses.

As in the IPython roadmap, we will indicate tasks by:

* Difficulty: Easy (E), Medium (M), Hard (H)
* Priority: 0 (most important), 1, 2, 3 (least important)
* Developer leading this task (doesn't have to be *sole* contributor,
but this is the point person to coordinate with if you want to contribute to
one of these tasks)

Eventually, we would like Flotilla to easily hook into and be hosted on an
Amazon EC2 server to perform computations and download memory-intensive
databases.

## Overview of 2014

Our work is being funded by the NumFOCUS foundation for July-December 2014.
With their support, we aim to release a stable, working product to users
early on, and from this release gain feature requests and bug reports for
edge cases we haven't thought of.

### Summer 2014

Currently, this project is private. Before we open this publicly, we need to
fix the following bugs:

* "Not" sample subset groups (e.g. "not neuron"): https://github.com/YeoLab/flotilla/issues/54
* Weird PC loadings: https://github.com/YeoLab/flotilla/issues/56

We plan to release a working version before August 15th,
2014. In addition to current functionality, this version ("0.2?") will have
the following spec:

* (E, 0) Naming issues. `experiment_design` data is apparently uninterpretable. 
Could go with `sample_metadata` or `phenotype_data` (as in 
[BioconductoR](http://www.bioconductor.org/)) instead.
    * https://github.com/YeoLab/flotilla/issues/47
* (M, 0) Outlier detection by expression or splicing (Boyko, Patrick, Olga)
    * Make this an interactive module that updates the metadata as you go
    along?
* (H, 0) Splicing modality detection (Olga)
    * Need to add some kind of confidence interval on the modality assignment
* (M, 0) Splicing
* (M, 0) Clustergram
    * Highly requested feature. Use `seaborn.clusteredheatmap`,
    when it is added.
* (E, 0) Easy data package creation
    * Using the [data packages](http://dataprotocols.org/data-packages/)
    specification from the Data Protocols people
* (H, 2) "Monocle" ordering of cells via psuedotime
    * May need to use `rpy2`? Or rewrite Monocle ourselves using `networkx`
    for the minimum spanning trees and such.
    * [Trapnell et al, *Nat Biotech* (2014)](http://www.nature
    .com/nbt/journal/v32/n4/full/nbt.2859.html)
* (H, 1) `DownsampledSplicingData`
    * Given data created by running MISO or other splicing algorithm,
    calculate log-log slopes for finding relationships between splicing
    events detected and sequencing depth.
* (E, 2) `Study`/`ExpressionData` should have option for log base,
e.g. if you have raw expresion data and want the `log10` transform. But we
still need to keep the original data.
* (E, 2) Refactor `Study` arguments to `expression_kws` and `splicing_kws`
because right now there's a whole mess of arguments and it's hard to
understand what's required and what's not.
* (M, 2) Add plots for `SpikeInData`
    * Violin plots of Spikein concentration vs expression value: Look at what
    concentration of molecules are detectable
    * Violin plots average distribution of spikein values within a cell
* (H, 1) Normalize to spikeins
    * Fit a `lowess` curve to spikein data across concentrations for each
    cell, then normalize each cell to its spikeins.
    * Also need support for weird configurations,
    e.g. in the same experiment, celltype A has only
    spikein X and celltype B has spikeins X and Y, but Y is bigger more
    reliable set. How do we use the information from celltype B to inform our
     normalization of celltype A?
* (H, 2) Identification of cellular subpopulations
    * As described in [Bruggner et al, *PNAS* (2014)](http://www.pnas
    .org/content/111/26/E2770.full)
* (E, 1) Security
    * Write up a document or disclaimer indicating to users that distributing
     their data this way is not completely secure. This is important for
     those working with clinical data shoe data storage and analysis
     tools must pass the United States' [HIPPA](http://en.wikipedia
     .org/wiki/Health_Insurance_Portability_and_Accountability_Act)
     regulations.
* (M, 1) Everything in [Fluidigm](http://www.fluidigm.com/home.html)'s
[SINGuLAR Analysis Toolset](http://www
.fluidigm.com/singular-analysis-toolset.html), including:
    * Outlier analysis
    * ANOVA
    * PCA
    * Hierarchical Clustering and heatmaps

After this release, we will add:

* (H, 1) Support for other species. Currently only have hand-curated datasets
 created for `hg19` and `mm10`.
    * Possibly through hooks into ENSEMBL/NCBI/other biological databases?
* (H, 1) Examples of real single-cell datasets analyzed through flotilla,
and all their figures re-created.
Candidate papers:
    * [Trapnell et al, *Nat Biotech* (2014)](http://www.nature
    .com/nbt/journal/v32/n4/full/nbt.2859.html)
    * [Shalek et al, *Nature* (2014)](http://www.nature
    .com/nature/journal/v510/n7505/full/nature13437.html)
    * [Patel et al, *Science* (2014)](http://www.sciencemag
    .org/content/344/6190/1396.abstract)

**Note: Abstract for [Biological Data Science](http://meetings.cshl
.edu/meetings/2014/data14.shtml) due August 22nd**

### Fall 2014

At this point, we aim to primarily use and distribute Flotilla on Amazon EC2
clusters.

* (H, 1) Integration with `pybedtools`, `pysam`, and possibly `metaseq` to
quickly grab conservation or genomic region information given gene names.
* (H, 3) DNA-seq analysis, as in the [SINGuLAR Analysis Toolset](http://www
.fluidigm.com/singular-analysis-toolset.html). Input is VCF files.
    * Variant quality and performance
    * Manhattan plots
    * Variant clustering

