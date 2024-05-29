Studies
=======

.. _TORQUE: https://adaptivecomputing.com/cherry-services/torque-resource-manager/
.. _hydra: https://hydra.cc/docs/intro/

To get a feeling for a data-driven experiment, two benchmark studies are available to run with the :mod:`f3dasm` package.
In order to run a study, you need to have the ``f3dasm[benchmark]`` extra requirements installed

 .. code-block:: console

    pip install f3dasm[benchmark]
     

Folder structure and files of a study
-------------------------------------

 .. code-block:: none
    :caption: Directory Structure

    ├── .
    │   └── my_study
    │       ├── main.py
    │       ├── config.yaml
    │       ├── pbsjob.sh
    │       └── README.md
    └── src/f3dasm

* Each study is put in a separate folder
* The `README.md` file gives a description, author and optionally citable source.
* The main script that has to be called should be named `main.py`
* `pbsjob.sh` is a batchscript file that will submit the `main.py` file to a `TORQUE`_ high-performance queuing system.
* The `config.yaml` are `hydra`_ configuration files.


Available studies
-----------------

There are two benchmark studies available:

+---------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
| Study                                                                                                                                       | Description                                                                |
+=============================================================================================================================================+============================================================================+
| `Fragile becomes supercompressible <https://github.com/bessagroup/f3dasm/tree/main/studies/fragile_becomes_supercompressible>`_             | Designing a supercompressible meta-material                                |
+---------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
| `Comparing optimization algorithms on benchmark functions <https://github.com/bessagroup/f3dasm/tree/pr/1.5/studies/benchmark_optimizers>`_ | Benchmark various optimization algorithms on analytical functions          |
+---------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------+
