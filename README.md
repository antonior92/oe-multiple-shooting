# oe-multiple-shooting

Identification of output error models using shooting methods.

## Instalation Guide

**This implementation only works in Python 3.**

1) **Download repository from github**

```bash
git clone git@github.com:antonior92/ip-nonlinear-solver.git
# or, alternatively:
# git clone https://github.com/antonior92/ip-nonlinear-solver.git
```

2) **Install package**

Move into the downloaded directory and install requirements with:
```bash
pip install -r requirements.txt
```

In sequence, install package with:
```bash
python setup.py install
```

3) **Test instalation**

The instalation can be tested by running:
```bash
python setup.py test
```
inside the downloaded directory.

## Reference

Both the implementation and the examples are originally from the paper:
```
"Shooting Methods for Parameter Estimation of Output Error Models" - IFAC World Congress (Toulouse, France - 2017). Antônio H. Ribeiro, Luis A. Aguirre.
```
available at IFAC-PapersOnLine ([here](http://www.sciencedirect.com/science/article/pii/S2405896317332469))

BibTeX entry:
```
@article{RIBEIRO201713998,
title = "Shooting Methods for Parameter Estimation of Output Error Models",
journal = "IFAC-PapersOnLine",
volume = "50",
number = "1",
pages = "13998 - 14003",
year = "2017",
issn = "2405-8963",
doi = "https://doi.org/10.1016/j.ifacol.2017.08.2421",
url = "http://www.sciencedirect.com/science/article/pii/S2405896317332469",
author = "Antônio H. Ribeiro and Luis A. Aguirre",
keywords = "Multiple shooting, output error models, simulation error minimization, nonlinear least-squares"
}
```


## Examples

Folder [``examples``](https://github.com/antonior92/oe-multiple-shooting/tree/master/examples) contain Jupyter notebooks for reproducing the examples presented in the paper.
