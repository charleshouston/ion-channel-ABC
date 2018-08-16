# ion_channel_ABC environment

This conda environment is required to run the myokit cell
simulations. To setup, first create an environment using
the spec file:

```
conda create -n ion_channel_ABC --file conda-envs/spec_ion_channel_ABC.txt
```

then activation the environment and pip install myokit from source:

```
conda activate ion_channel_ABC
pip install --explicit=git+https://github.com/MichaelClerx/myokit.git@master#egg=myokit
```

# pyabc environment

This environment is used to run jupyter notebooks and contains the library
for the core ABC algorithm used. To setup, first create the environment
using the supplied spec file:

```
conda create -n pyabc --file conda-envs/spec_pyabc.txt
```

then download the source for pyabc and apply to patches supplied.
Finally, pip install the library in develop mode:

```
pip install -e .
```
(from within the patched pyabc directory).
