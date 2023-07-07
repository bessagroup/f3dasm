Design
------

Each set of input and output parameters in the ExperimentData class is called a `Design`
The `Design` can be used in your own scripts and function to specify the design variables.

.. code-block:: python
    
   from f3dasm import Design

    def my_function(design: Design):
        parameter1 = design.get('parameter1')
        parameter2 = design.get('parameter2')

        ...  # Your own program

        design.set('output1', output)
        return design


A function like `my_function` can be used in the `ExperimentData.run()` function to iterate over every design in the ExperimentData.