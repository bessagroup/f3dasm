.. _workflow:

Create a workflow
=================

A workflow is the pipeline function that determines the steps in order to create of the data-driven process. 

All of the :mod:`f3dasm` classes come together in the workflow script. It serves as the entry-point of your data-driven application.

The worfklow should only act on the interfaces of the :mod:`f3dasm` classes and any experiment-specific scripts can be imported into the workflow.
In this way, the workflow itself is independent of the experiment-specific scripts and can be reused for different experiments.

Example
-------

In the following examples, we will create a workflow for the following data-driven process:

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
- `main.py` is the main entry point of the project, governed by :mod:`f3dasm`.
- `my_script.py` contains the user-defined script. In this case a custom data-generationr function `my_function`.

.. code-block:: none
   :caption: Directory Structure

   my_project/
   ├── my_script.py
   └── main.py


.. _my-script:

my_script.py
^^^^^^^^^^^^

The `my_script.py` file contains your own `my_function` function. You have to modify the function so that it conforms with the :class:`~f3dasm.datageneration.datagenerator.DataGenerator` interface. 

.. note::
    Learn more at the section :ref:`data-generation-function` on how to comply.

Because we are using the :class:`~f3dasm.datageneration.functions.pybenchfunction.Ackley` function, this function is already compliant with the interface and we do not need to modify it:

.. code-block:: python
   :caption: my_script.py

    from f3dasm import Design

    def my_function(design: Design, benchmark_function) -> Design:
        return benchmark_function(design)

.. note::
    As shown in the code snippet above, the :class:`~f3dasm.design.design.Design` object is the only connector between your scripts and the :mod:`f3dasm` interface.
    Therefore, you can use any package or third-party software call you want in your scripts, as long as you can pass the :class:`~f3dasm.design.design.Design` object to the function and return it back.

main.py
^^^^^^^

The `main.py` file is the main entry point of the project. It contains the :mod:`f3dasm` classes and acts on these interfaces.
It imports :mod:`f3dasm` and the `my_function` from `my_script.py`. 
In the main function, we create the :class:`~f3dasm.design.domain.Domain`, sample from the :class:`~f3dasm.sampling.latinhypercube.LatinHypercube` sampler , and executes the data generation function (`my_function`) using the :meth:`~f3dasm.design.experimentdata.Experiment.run` method with the specified execution mode.

.. code-block:: python
   :caption: main.py

    from f3dasm.sampling import LatinHypercube
    from f3dasm.design import make_nd_continuous_domain
    from f3dasm.datageneration.functions import Ackley
    from f3dasm.optimization import LBFGSB
    from my_script import my_function

    """Design of Experiment"""
    # Create a domain object
    domain = f3dasm.design.make_nd_continuous_domain(bounds=np.tile([0.0, 1.0], (20, 1)), dimensionality=20)

    # Sampling from the domain
    sampler = f3dasm.sampling.LatinHypercube(domain)
    data = sampler.get_samples(numsamples=10)

    """Data Generation"""
    # Initialize the simulator
    ackley_function = Ackley(dimensionality=20, bounds=domain.get_bounds())

    # Use the data-generator to evaluate the initial samples
    data.run(my_function, mode='sequential', kwargs={'benchmark_function': ackley_function)


    """Optimization"""
    optimizer = LBFGSB(data)
    optimizer.iterate(100, my_function, mode='sequential', kwargs={'benchmark_function': ackley_function})

    # Extract and store the optimization results
    optimized_data = optimizer.extract_data()
    optimized_data.store()

.. note::
    In the `main.py` file, notice that there is only one connection with the `my_script.py` file, which is the `my_function` function import. 
    This means that the workflow file (`main.py`) is independent of the application file (`my_script.py`) and can be reused for different experiments.

