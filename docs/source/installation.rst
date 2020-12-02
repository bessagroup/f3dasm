Installation guide
==================

.. |f3dasm| replace:: F3DASM
.. _f3dasm: https://github.com/bessagroup/F3DASM>

.. |abaqus| replace:: Abaqus
.. _abaqus: https://www.3ds.com/products-services/simulia/products/abaqus/




|f3dasm|_ can be installed as any other Python package via ``pip``:



.. code-block::

   >> pip install f3dasm


.. warning::
   Most of |f3dasm|_ code requires access to |abaqus|_. Please, ensure you have available licenses before you launch your simulations.


In order to protect your Python installation, you can create a virtual environment where to install |f3dasm|_ and all its dependencies. Among several options, you can create a ``conda environment``:

.. code-block::

   >> conda create -n f3dasm_env python


When |f3dasm|_ is installed via ``pip``, all the dependencies are also installed. Nevertheless, some examples require access to additional libraries, such as ``jupyter`` and ``scipy``. You can easily install them while you create your virtual environment:

.. code-block::

   >> conda create -n f3dasm_env jupyter scipy



.. tip::
    Install ``nb_conda`` to ensure jupyter notebooks work properly within virtual environments.
