Design of Flotilla
==================

* Vision: have the single-cell project that took 6 months happen in 2 months.
    * How: Offload interpetation to someone who actually did the experiment
    * Reproducibility: Take the whole project from one notebook or person to
    another.
        * Q: How to rewrite/memoize/cache the metadata?

Target Audience
---------------

The wet lab

Example work session
--------------------

Required: How to create a new project

    flotilla make_project \
        --project-name super_awesome_project \
        --expression [expression.tsv] \
        --splicing [splicing.tsv] \
        --metadata [metadata.tsv]

Test that the project was made correctly:

    flotilla test_project --project-name super_awesome_project

Go into the directory:

    cd super_awesome_project

Serve up an IPython notebook:

    flotilla start_notebook neural_diff_project

"Mix" related projects (e.g. neural diff single cell and cardiac diff single
cell):

    flotilla mix_projects

Access data that's out there:

* Encode
* Brainspan
* Bodymap

### IPython notebook analysis steps

    import cardiac_project
    import flotilla

    cardiac = flotilla.embark(cardiac_project)
    cardiac.pca()
    cardiac.monocle()
    cardiac.splicing.modalities()

    import neural_diff_project

    neural_diff = flotilla.embark([])

Next steps
----------

* Rewrite `_*BaseData.py` to have `BaseData` objects which inherit from `pandas.DataFrame`
    * Rename `BaseData` --> `BaseData` to be explicit that it's a Base class and
    can't do anything
    * Look into efficient memoization/cacheing for storing results of
    PCA/JSD/NMF so don't need to be calculated again.
        * a database potentially?

