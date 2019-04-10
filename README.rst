ion-channel-ABC
===============

Calibrate cardiac electrophysiology cell models using Approximate
Bayesian Computation with the myokit_ and pyabc_ libraries.

Installation
------------

#. Create local clone of this repository.
#. **Recommended** Create new python environment using miniconda_. The
   environment should be created for Python 3+ and with Jupyter notebook
   support to be able to run the example notebooks, e.g. ``conda create -n
   ionchannelABC python=3 jupyter``.
#. Activate the previously created miniconda environment, e.g. ``conda activate
   ionchannelABC``.
#. Install pyabc library (for likelihood-free inference) and myokit library
   (for running electrophysiology cell models) using pip, e.g. ``pip install
   myokit pyabc``.
#. Update the paths to a SUNDIALS installation in the ``myokit.ini`` file which
   is created on the user's home path ``~/.config/myokit/myokit.ini``.
#. If not already existing, create an environment variable to a temporary
   directory ``TMPDIR`` necessary for myokit to save local files. A line could
   be added to your ``~/.bashrc``: ``export TMPDIR=/path/to/tmp/directory``.
#. Finally, install the ion-channel-ABC package by navigating to your cloned
   repository and running ``python setup.py install``.

Running
-------

Example Jupyter notebooks demonstrating use of key features are available in
(docs/examples) folder. It is recommended to start with the
getting_started.ipynb_ notebook.

.. _myokit: http://myokit.org
.. _pyabc: https://github.com/icb-dcm/pyabc
.. _miniconda: https://conda.io/miniconda.html
.. _getting_started.ipynb: docs/examples/getting_started.ipynb
