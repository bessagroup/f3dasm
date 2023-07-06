Create a workflow
-----------------

Example
```````

Directory Structure:
====================

The directory structure for the project is as follows:

- `my_project/` is the root directory.
- `my_script.py` contains the implementation of the `my_function` function.
- `main.py` is the main entry point of the project, governed by `f3dasm`.

.. code-block:: none
   :caption: Directory Structure

   my_project/
   ├── my_script.py
   └── main.py


my_script.py
=============

The `my_script.py` file contains your own `my_function` function. You have to modify the function so that it takes a `f3dasm.design` object as input.
retrieves the values of `parameter1` and `parameter2` from the design, performs some calculations or operations, and sets the value of `output1` in the design. 
The function has to return the (modified) design object.


.. code-block:: python
   :caption: my_script.py

    def my_function(design):
        parameter1 = design.get('parameter1')
        parameter2 = design.get('parameter2')
        ...

        design.set('output1', output)
        return design


main.py
========

The `main.py` file is the main entry point of the project. 
It imports `f3dasm` and the `my_function` from `my_script.py`. 
In the main function, it creates a design space, fills the design space using a sampler, and executes the data generation function (`my_function`) using the `data.run` method with the specified execution mode.

.. code-block:: python
   :caption: main.py

    import f3dasm
    from my_script import my_function

    """Block 1: Design of Experiment"""

    # Create a design space
    design = f3dasm.Domain()

    design.add_input_space(name="parameter1", space=f3dasm.ContinuousParameter(
        lower_bound=0.0, upper_bound=1.0))
    design.add_input_space(name="parameter2", space=f3dasm.ContinuousParameter(
        lower_bound=0.0, upper_bound=1.0))

    design.add_output_space(name="output1", space=f3dasm.ContinuousParameter())

    # Filling the design space
    sampler = f3dasm.sampling.RandomUniform(design)
    data = sampler.get_samples(numsamples=3)

    """Block 2: Data Generation"""

    # Execute the data generation function
    data.run(my_function, mode='sequential')


    # Store the data generation function
    data.store()



