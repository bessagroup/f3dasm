Getting Started
===============

The installation consists of two steps

* Installing and configuring miniconda
* Installing :code:`f3dasm`

Installing and configuring miniconda
------------------------------------

Download `Miniconda <https://docs.conda.io/en/latest/miniconda.html#linux-installers>`_ and add :code:`conda-forge` to the channels:

.. code-block:: console

  $ conda config --add channels conda-forge

Installing :code:`f3dasm`
-------------------------

Clone from the GitHub repository

.. code-block:: console

  $ git clone https://github.com/bessagroup/F3DASM.git --branch v1.0.0 --single-branch

Preferred way
^^^^^^^^^^^^^

Create a new environment from the :code:`f3dasm_environment.yml` file

.. code-block:: console

  $ cd F3DASM
  $ conda env create -f f3dasm_environment.yml

Test if the installation was successful

.. code-block:: console

  $ conda activate f3dasm_env
  $ make test-smoke

If the smoke tests pass the installation is successful!
Now install the package in editable mode:

.. code-block:: console

  $ pip install -e .

You can now use :code:`import f3dasm`

.. code-block:: console

  $ python
  >>> import f3dasm

If no errors occur when importing the package, then you have succesfully installed the :code:`f3dasm` package!

If things fail
^^^^^^^^^^^^^^

You can also create the required conda environment from scratch.
Create a new python 3.10 environment

.. code-block:: console

  $ cd F3DASM
  $ conda create -n f3dasm_env python=3.10

Now install the package in editable mode:

.. code-block:: console

  $ pip install -e .

You can now use :code:`import f3dasm`

.. code-block:: console

  $ python
  >>> import f3dasm

If no errors occur when importing the package, then you have succesfully installed the :code:`f3dasm` package!

Installing all packages manually
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a last resort, you could also create a new python 3.10 environment and install the required packages directly:

.. code-block:: console

  $ cd F3DASM
  $ conda create -n f3dasm_env python=3.10
  $ pip install -e .
  $ conda install autograd numdifftools tensorflow pygmo pathos pytest pytest-cov
  $ make test-smoke
