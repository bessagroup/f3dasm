Getting Started
===============


Install from source
-------------------

Download `Miniconda <https://docs.conda.io/en/latest/miniconda.html#linux-installers>`_ and add :code:`conda-forge` to the channels and set the priority to strict:

.. code-block:: console

  $ conda config --add channels conda-forge
  $ conda config --set channel_priority strict

Clone from the GitHub repository

.. code-block:: console

  $ git clone https://github.com/mpvanderschelling/testthings.git


Create a new environment from the :code:`f3dasm_environment.yml` file

.. code-block:: console

  $ cd testthings
  $ conda env create -f f3dasm_environment.yml

Test if the installation was successful

.. code-block:: console

  $ conda activate f3dasm_env
  $ make test-smoke

If the tests pass the installation is successful!
Now install the package in editable mode:

.. code-block:: console

  $ pip install -e .

You can now use :code:`import f3dasm`

.. code-block:: console

  $ python
  >>> import f3dasm