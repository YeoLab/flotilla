flotilla
========
download with:
```
git clone --recursive https://github.com/YeoLab/flotilla.git
```
and download the singlecell project (for testing porpoises) with:
```
git clone https://github.com/YeoLab/neural_diff_project.git
```

build/install with:

note: for some reason patsy isn't installing automatically with pip, use easy_install first instead
```
easy_install -U patsy
cd flotilla
pip install .
cd ..
cd neural_diff_project
pip install -e .
cd ..
```
start a notebook
```
serve_flotilla_notebook neural_diff_project/notebook
```


check intro to flotila.html/ipynb for instructions