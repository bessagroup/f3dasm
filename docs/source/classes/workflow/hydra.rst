Integration with Hydra
----------------------

Example
```````

Directory Structure:
====================

The directory structure for the project is as follows:

- `my_project/` is the root directory.
- `my_script.py` contains the implementation of the `my_function` function.
- `config.yaml` is a YAML configuration file.
- `main.py` is the main entry point of the project.

.. code-block:: none
   :caption: Directory Structure

   my_project/
   ├── my_script.py
   ├── config.yaml
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


config.yaml
============

The `config.yaml` file contains the configuration settings for the project. 
It specifies the design space, input and output spaces, sampler settings, and the execution mode (`sequential` in this case).
The design space is defined with `parameter1` and `parameter2` as continuous parameters with their corresponding lower and upper bounds.


.. code-block:: yaml
   :caption: config.yaml

    design:
        input_space:
            parameter1:
                _target_: f3dasm.ContinuousParameter
                lower_bound: 0.0
                upper_bound: 1.0
            parameter2:
                _target_: f3dasm.ContinuousParameter
                lower_bound: 0.0
                upper_bound: 1.0

        output_space:
            output1:
                _target_: f3dasm.ContinuousParameter

        sampler:
            _target_: f3dasm.sampling.RandomUniform
            seed: 1
            number_of_samples: 3

        mode: sequential


main.py
========

The `main.py` file is the main entry point of the project. 
It imports necessary modules (`f3dasm`, `hydra`) and the `my_function` from `my_script.py`. 
The script defines a `main` function decorated with `@hydra.main`, which reads the configuration from `config.yaml`. 
In the main function, it creates a design space, fills the design space using a sampler, and executes the data generation function (`my_function`) using the `data.run` method with the specified execution mode.



.. code-block:: python
   :caption: main.py

    import f3dasm
    import hydra
    from my_script import my_function

    @hydra.main(config_path=".", config_name="config")
    def main(config):
        """Block 1: Design of Experiment"""

        # Create a design space
        design = f3dasm.Domain.from_yaml(config.design)

        # Filling the design space
        sampler = f3dasm.sampling.Sampler.from_yaml(config)
        data = f3dasm.ExperimentData.from_sampling(sampler)

        """Block 2: Data Generation"""

        # Execute the data generation function
        data.run(my_function, mode=config.experimentdata.mode)

        # Store the data to a file
        data.store()

    if __name__ == "__main__":
        main()

