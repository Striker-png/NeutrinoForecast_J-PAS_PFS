# J-PAS and PFS surveys in the era of dark energy and neutrino mass measurements

This repository stores all the code and data needed to reproduce the results of the Fisher forecast on the sum of neutrino masses and dark energy for J-PAS and PFS ([arXiv:2505.04275](https://arxiv.org/abs/2505.04275), or [Fuxing Qin *et al* 2026 *ApJ* **997** 251](https://iopscience.iop.org/article/10.3847/1538-4357/ae261e)). 

The survey specifications, forecast settings and data for various cases can be found in the `code` folder. 
For forecast on dynamical dark energy using a nonfiducial cosmology, they are stored in the `dynamical_DE_nonfid` folder.

If you want to run the code, you have to install the `starfish` package first. 
`starfish` containts the whole pipeline for the Fisher forecast used in this paper.
And it can be reused for forecast of other surveys, if different `settings.py` files are provided.

## Installation

Requirements: `numpy`, `scipy`, `camb`, `matplolib`, `pandas`, `astropy`.

Simply run:
```
pip install ./starfish
```

`starfish` works well for Python version 3.12.3 on the author's laptop, but it is not tested for lower versions.

The physical modeling behind the code can be found in the paper.
