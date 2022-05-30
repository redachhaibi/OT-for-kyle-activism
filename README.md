# OT-for-kyle-activism
Illustration of the Optimal Transport for the Kyle Model

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- ipynb: Contains Python notebooks which demonstrate how the code works
|  |-- OT-for-kyle-activism.ipynb: Makes the plots.
|-- ot_for_kyle_activism:
   |-- core.py: Main routines, to be imported as external module
|-- LICENSE  : MIT LICENSE
|-- README.md: This file
|-- setup.py : Setup file for install
```

## Installation

1. Create new virtual environment

```bash
$ python3 -m venv .venv_ot_for_kyle
```

(Do
sudo apt install python3-venv
if needed)

3. Activate virtual environment

```bash
$ source .venv_ot_for_kyle/bin/activate
```

4. Upgrade pip, wheel and setuptools 

```bash
$ pip install --upgrade pip
$ pip install --upgrade setuptools
$ pip install wheel
```

5. Install the package.

```bash
python setup.py develop
```

6. (Optional) In order to use Jupyter with this virtual environment .venv
```bash
pip install jupyter
pip install --user ipykernel
python -m ipykernel install --user --name=.venv_ot_for_kyle
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

## Configuration
Nothing to do

## Credits
Later