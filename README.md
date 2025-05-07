# J-PAS and PFS surveys in the era of dark energy and neutrino mass measurements

This repository stores all the code and data needed to reproduce the results of the Fisher forecast on the sum of neutrino masses and dark energy for J-PAS and PFS. 
The survey specifications, forecast settings and data for various cases can be found in the `code` folder. 
For forecast on dynamical dark energy using a nonfiducial cosmology, they are stored in the `dynamical_DE_nonfid` folder.

If you want to run the code, you have to install the `starfish` package first. 
`starfish` containts the whole pipeline for the Fisher forecast used in this paper.
And it can be reused for forecast of other surveys, if different `settings.py` files provided.

## Installation

Requirements: `numpy`, `scipy`, `camb`, `matplolib`, `pandas`, `astropy`.

Run:
```
cd ./starfish/
python setup.py sdist
pip install ./dist/starfish-0.1.0.tar.gz
```

`starfish` works well for Python version 3.12.3 on the author's laptop, but is not tested for lower versions.

The physical modeling behind the code can be found in the paper.