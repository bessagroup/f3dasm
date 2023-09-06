.. _hydra: https://hydra.cc/

Integration with `hydra`_
=========================

`hydra`_ is an open-source configuration management framework that is widely used in machine learning and other software development domains.
It is designed to help developers manage and organize complex configuration settings for their projects, 
making it easier to experiment with different configurations, manage multiple environments, and maintain reproducibility in their work.

`hydra`_ can be seamlessly integrated with the worfklows in :mod:`f3dasm` to manage the configuration settings for the project.

Example
-------

The following example is the same as in section :ref:`workflow`; we will create a workflow for the following data-driven process:

* Create a 20D continuous :class:`~f3dasm.design.domain.Domain`
* Sample from the domain using a the :class:`~f3dasm.sampling.latinhypercube.LatinHypercube` sampler
* Use a data generation function, which will be the :class:`~f3dasm.datageneration.functions.pybenchfunction.Ackley` function a from the :ref:`benchmark-functions`
* Optimize the data generation function using the built-in :class:`~f3dasm.optimization.lbfgsb.LBFGSB` optimizer.

.. image:: ../../../img/f3dasm-workflow-example.png
   :width: 70%
   :align: center
   :alt: Workflow

|

Directory Structure
^^^^^^^^^^^^^^^^^^^

The directory structure for the project is as follows:

- `my_project/` is the root directory.
- `my_script.py` contains the user-defined script. In this case a custom data-generationr function `my_function`.
- `config.yaml` is a hydraYAML configuration file.
- `main.py` is the main entry point of the project, governed by :mod:`f3dasm`.


.. code-block:: none
   :caption: Directory Structure

   my_project/
   ├── my_script.py
   ├── config.yaml
   └── main.py


my_script.py
^^^^^^^^^^^^

The user-defined script is identical to the one in :ref:`my-script`.

config.yaml
^^^^^^^^^^^

The `config.yaml` file contains the configuration settings for the project. 
You can create configurations for each of the :mod:`f3dasm` classes:

============================================================= ======================================================
Class                                                         Section referencing how to create the `hydra`_ config            
============================================================= ======================================================
:class:`~f3dasm.design.domain.Domain`                         :ref:`domain-from-yaml`         
:class:`~f3dasm.sampling.Sampler`                             :ref:`sampler-hydra`  
:class:`~f3dasm.design.experimentdata.ExperimentData`         :ref:`experimentdata-hydra`
:class:`~f3dasm.optimization.optimizer.Optimizer`             to be implemented!
:class:`~f3dasm.datageneration.datagenerator.DataGenerator`   to be implemented!
============================================================= ======================================================



.. code-block:: yaml
   :caption: config.yaml

    domain:
        input_space:
            x0:
                _target_: f3dasm.ContinuousParameter
                lower_bound: 0.0
                upper_bound: 1.0
            x1:
                _target_: f3dasm.ContinuousParameter
                lower_bound: 0.0
                upper_bound: 1.0

    sampler:
        _target_: f3dasm.sampling.LatinHypercube
        seed: 1
        number_of_samples: 3

    experimentdata:
        from_sampling: ${sampler}

    mode: sequential

It specifies the search-space domain, sampler settings, and the execution mode (`sequential` in this case).
The domain is defined with `x0` and `x1` as continuous parameters with their corresponding lower and upper bounds.

main.py
^^^^^^^

The `main.py` file is the main entry point of the project. It contains the :mod:`f3dasm` classes and acts on these interfaces.
It imports :mod:`f3dasm` and the `my_function` from `my_script.py`. 


The `main.py` file is the main entry point of the project. 

* It imports the necessary modules (`f3dasm`, `hydra`) and the `my_function` from `my_script.py`. 
* Inside `main.py` script defines a :code:`main` function decorated with :code:`@hydra.main`, which reads the configuration from :code:`config.yaml`. 
* Within the :code:`main` function, we instantiate the :class:`~f3dasm.design.domain.Domain`, sample from the :class:`~f3dasm.sampling.latinhypercube.LatinHypercube` sampler , and executes the data generation function (`my_function`) using the :meth:`~f3dasm.design.experimentdata.Experiment.run` method with the specified execution mode.



.. code-block:: python
   :caption: main.py

    from f3dasm.design import ExperimentData
    from f3dasm.datageneration.functions import Ackley
    from f3dasm.optimization import LBFGSB
    from my_script import my_function

    @hydra.main(config_path=".", config_name="config")
    def main(config):    
        """Design of Experiment"""
        # Create a domain object
        domain = f3dasm.Domain.from_yaml(config.domain)

        # Sampling from the domain
        data = f3dasm.ExperimentData.from_yaml(config)

        """Data Generation"""
        # Initialize the simulator
        ackley_function = Ackley(dimensionality=20, bounds=domain.get_bounds())

        # Use the data-generator to evaluate the initial samples
        data.run(my_function, mode=config.mode, kwargs={'benchmark_function': ackley_function)

        """Optimization"""
        optimizer = LBFGSB(data)
        optimizer.iterate(100, my_function, mode=config.mode, kwargs={'benchmark_function': ackley_function})

        # Extract and store the optimization results
        optimized_data = optimizer.extract_data()
        optimized_data.store()

    if __name__ == "__main__":
        main()

.. note::
    To use `hydra`_ on a high-performance computing cluster, take a look at the :ref:`hydra-on-hpc` section.
