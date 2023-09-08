.. _benchmark-functions:

Benchmark functions
===================

:mod:`f3dasm` comes with a set of benchmark functions that can be used to test the performance of 
optimization algorithms or to mock some expensive simulation in order to test the workflow

Usage
-----

Benchmarkfunction are a submodule of the :code:`f3dasm.datageneration` module under the name :code:`functions`

.. code-block:: python

  from f3dasm.datageration.functions import Sphere

Creating the function
^^^^^^^^^^^^^^^^^^^^^

First we make a 2-dimensional continous domain with the helper function :func:`~f3dasm.design.design.make_nd_continuous_design`:

.. code-block:: python

  bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
  domain = f3dasm.make_nd_continuous_design(bounds=bounds, dimensions=2)

Then we create an object of the :class:`~f3dasm.functions.pybenchfunction.Sphere` class, specifying the :attr:`~f3dasm.functions.pybenchfunction.PybenchFunction.dimensionality`:

.. code-block:: python
 
  sphere_function = Sphere(dimensionality=dim, scale_bounds=bounds)

Evaluating the function
^^^^^^^^^^^^^^^^^^^^^^^

The function can be evaluated by directly calling the object

.. code-block:: python

  x = np.array([0.5, 0.8])
  >>> sphere_function(x)
  ... array([[23.330816]])

Gradient
^^^^^^^^

The gradient of a function can be estimated by calling the :func:`~f3dasm.base.function.Function.dfdx` function:

.. code-block:: python

  >>> sphere_function.dfdx(x)
  ... array([[26.2144 , 41.94304]])
  
The gradient is calculated with the :code:`autograd` package.

Plotting
^^^^^^^^

We can plot a 3D or 2D representation of the loss landscape:

.. code-block:: python

  sphere_function.plot(orientation="3D", domain=bounds)
  
img


.. code-block:: python

  sphere_function.plot(orientation="2D", domain=bounds)
  
img


Augmentor
---------

In order to further diversify your benchmark functions, it is possible to add add data augmentation to you benchmark functions.
Within :mod:`f3dasm` this is done with the :class:`~f3dasm.functions.adapters.augmentor.Augmentor` class.
The following three augmentation operations are supported in :mod:`f3dasm`:

- :class:`~f3dasm.functions.adapters.augmentor.Scale`: Scaling the boundaries of the function to another set of lower and upper boundaries
- :class:`~f3dasm.functions.adapters.augmentor.Offset`: Offsetting the benchmarkfunction by a constant vector
- :class:`~f3dasm.functions.adapters.augmentor.Noise`: Adding Gaussian noise to the objective value.

You can create any combination of augmentors and supply them in lists to create a :class:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor` object.

- You can add a list of augmentors that work on the **input vector** to the :attr:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.input_augmentors` attribute with the :meth:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.add_input_augmentor` method.
- You can add a list of augmentors that work on the **objective value** to the :attr:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.output_augmentors` attribute with the :meth:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.add_output_augmentor` method.

Whenever you evaluate the benchmark function, the input and output vectors will be manipulated by the augmentors in the :class:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor` in order.
You can retrieve the original value from a vector that has been manipulated by the augmentors by calling the :meth:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor.augment_reverse_input` method.

When a benchmarkfunction object is created, an empty :class:`~f3dasm.functions.adapters.augmentor.FunctionAugmentor` is created and stored as attribute (:class:`~f3dasm.functions.Function.augmentor`). 
If you provide one of the following initialization attributes to the object, augmentors are created and added accordingly:

- :attr:`~f3dasm.functions.adapters.pybenchfunction.PyBenchFunction.scale_bounds`, if set not to None
- :attr:`~f3dasm.functions.adapters.pybenchfunction.PyBenchFunction.offset` if set to True, (default value is True)
- :attr:`~f3dasm.functions.adapters.pybenchfunction.PyBenchFunction.noise` if set not to None

Create your own augmentor
^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create your own augmentor, create a new class and inheret from the base :class:`~f3dasm.functions.adapters.augmentor.Augmentor` class:

.. code-block:: python

  class NewAugmentor(Augmentor):
      """
      Base class for operations that augment an loss-funciton
      """
  
      def augment(self, input: np.ndarray) -> np.ndarray:
          ...
  
      def reverse_augment(self, output: np.ndarray) -> np.ndarray:
          ...



Implemented benchmark functions
-------------------------------

The following implementations of benchmark functions can found under the :mod:`f3dasm.functions` module.
These are taken and modified from the `Python Benchmark Test Optimization Function Single Objective <https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective>`_ github repository.

Convex functions
^^^^^^^^^^^^^^^^

======================== ====================================================================================
Name                      Docs of the Python class                                              
======================== ====================================================================================
Ackley N. 2              :class:`~f3dasm.datageneration.functions.pybenchfunction.AckleyN2`
Bohachevsky N. 1         :class:`~f3dasm.datageneration.functions.pybenchfunction.BohachevskyN1`
Booth                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Booth`
Brent                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Brent`
Brown                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Brown`
Bukin N. 6               :class:`~f3dasm.datageneration.functions.pybenchfunction.BukinN6`
Dixon Price              :class:`~f3dasm.datageneration.functions.pybenchfunction.DixonPrice`
Exponential              :class:`~f3dasm.datageneration.functions.pybenchfunction.Exponential`
Matyas                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Matyas`
McCormick                :class:`~f3dasm.datageneration.functions.pybenchfunction.McCormick`
Perm 0, d, beta          :class:`~f3dasm.datageneration.functions.pybenchfunction.PermZeroDBeta`
Powell                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Powell`
Rotated Hyper-Ellipsoid  :class:`~f3dasm.datageneration.functions.pybenchfunction.RotatedHyperEllipsoid`
Schwefel 2.20            :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel2_20`
Schwefel 2.21            :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel2_21`
Schwefel 2.22            :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel2_22`
Schwefel 2.23            :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel2_23`
Sphere                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Sphere`
Sum Squares              :class:`~f3dasm.datageneration.functions.pybenchfunction.SumSquares`
Thevenot                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Thevenot`
Trid                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Trid`
Xin She Yang N.3         :class:`~f3dasm.datageneration.functions.pybenchfunction.XinSheYangN3`
Xin-She Yang N.4         :class:`~f3dasm.datageneration.functions.pybenchfunction.XinSheYangN4`
======================== ====================================================================================


Seperable functions
^^^^^^^^^^^^^^^^^^^

======================== ====================================================================================
Name                      Docs of the Python class                                              
======================== ====================================================================================
Ackley                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Ackley`
Bohachevsky N. 1         :class:`~f3dasm.datageneration.functions.pybenchfunction.BohachevskyN1`
Easom                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Easom`
Egg Crate                :class:`~f3dasm.datageneration.functions.pybenchfunction.EggCrate`
Exponential              :class:`~f3dasm.datageneration.functions.pybenchfunction.Exponential`
Griewank                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Griewank`
Michalewicz              :class:`~f3dasm.datageneration.functions.pybenchfunction.Michalewicz`
Powell                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Powell`
Qing                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Qing`
Quartic                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Quartic`
Rastrigin                :class:`~f3dasm.datageneration.functions.pybenchfunction.Rastrigin`
Schwefel                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel`
Schwefel 2.20            :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel2_20`
Schwefel 2.21            :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel2_21`
Schwefel 2.22            :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel2_22`
Schwefel 2.23            :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel2_23`
Sphere                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Sphere`
Styblinski Tank          :class:`~f3dasm.datageneration.functions.pybenchfunction.StyblinskiTank`
Sum Squares              :class:`~f3dasm.datageneration.functions.pybenchfunction.SumSquares`
Thevenot                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Thevenot`
Xin She Yang             :class:`~f3dasm.datageneration.functions.pybenchfunction.XinSheYang`
======================== ====================================================================================


Differentiable functions
^^^^^^^^^^^^^^^^^^^^^^^^

======================== ====================================================================================
Name                      Docs of the Python class                                              
======================== ====================================================================================
Ackley                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Ackley`
Ackley N. 2              :class:`~f3dasm.datageneration.functions.pybenchfunction.AckleyN2`
Ackley N. 3              :class:`~f3dasm.datageneration.functions.pybenchfunction.AckleyN3`
Ackley N. 4              :class:`~f3dasm.datageneration.functions.pybenchfunction.AckleyN4`
Adjiman                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Adjiman`
Beale                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Beale`
Bird                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Bird`
Bohachevsky N. 1         :class:`~f3dasm.datageneration.functions.pybenchfunction.BohachevskyN1`
Bohachevsky N. 2         :class:`~f3dasm.datageneration.functions.pybenchfunction.BohachevskyN2`
Bohachevsky N. 3         :class:`~f3dasm.datageneration.functions.pybenchfunction.BohachevskyN3`
Booth                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Booth`
Branin                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Branin`
Brent                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Brent`
Brown                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Brown`
Colville                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Colville`
De Jong N. 5             :class:`~f3dasm.datageneration.functions.pybenchfunction.DeJongN5`
Deckkers-Aarts           :class:`~f3dasm.datageneration.functions.pybenchfunction.DeckkersAarts`
Dixon Price              :class:`~f3dasm.datageneration.functions.pybenchfunction.DixonPrice`
Drop-Wave                :class:`~f3dasm.datageneration.functions.pybenchfunction.DropWave`
Easom                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Easom`
Egg Crate                :class:`~f3dasm.datageneration.functions.pybenchfunction.EggCrate`
Egg Holder               :class:`~f3dasm.datageneration.functions.pybenchfunction.EggHolder`
Exponential              :class:`~f3dasm.datageneration.functions.pybenchfunction.Exponential`
Goldstein-Price          :class:`~f3dasm.datageneration.functions.pybenchfunction.GoldsteinPrice`
Griewank                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Griewank`
Happy Cat                :class:`~f3dasm.datageneration.functions.pybenchfunction.HappyCat`
Himmelblau               :class:`~f3dasm.datageneration.functions.pybenchfunction.Himmelblau`
Keane                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Keane`
Langermann               :class:`~f3dasm.datageneration.functions.pybenchfunction.Langermann`
Leon                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Leon`
Levy                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Levy`
Levy N. 13               :class:`~f3dasm.datageneration.functions.pybenchfunction.LevyN13`
Matyas                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Matyas`
McCormick                :class:`~f3dasm.datageneration.functions.pybenchfunction.McCormick`
Michalewicz              :class:`~f3dasm.datageneration.functions.pybenchfunction.Michalewicz`
Periodic                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Periodic`
Perm d, beta             :class:`~f3dasm.datageneration.functions.pybenchfunction.PermDBeta`
Perm 0, d, beta          :class:`~f3dasm.datageneration.functions.pybenchfunction.PermZeroDBeta`
Qing                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Qing`
Quartic                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Quartic`
Rastrigin                :class:`~f3dasm.datageneration.functions.pybenchfunction.Rastrigin`
Ridge                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Ridge`
Rosenbrock               :class:`~f3dasm.datageneration.functions.pybenchfunction.Rosenbrock`
Rotated Hyper-Ellipsoid  :class:`~f3dasm.datageneration.functions.pybenchfunction.RotatedHyperEllipsoid`
Salomon                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Salomon`
Schaffel N. 1            :class:`~f3dasm.datageneration.functions.pybenchfunction.SchaffelN1`
Schaffel N. 2            :class:`~f3dasm.datageneration.functions.pybenchfunction.SchaffelN2`
Schaffel N. 3            :class:`~f3dasm.datageneration.functions.pybenchfunction.SchaffelN3`
Schaffel N. 4            :class:`~f3dasm.datageneration.functions.pybenchfunction.SchaffelN4`
Shekel                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Shekel`
Shubert                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Shubert`
Shubert N. 3             :class:`~f3dasm.datageneration.functions.pybenchfunction.ShubertN3`
Shubert N. 4             :class:`~f3dasm.datageneration.functions.pybenchfunction.ShubertN4`
Styblinski Tank          :class:`~f3dasm.datageneration.functions.pybenchfunction.StyblinskiTank`
Sum Squares              :class:`~f3dasm.datageneration.functions.pybenchfunction.SumSquares`
Thevenot                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Thevenot`
Three-Hump               :class:`~f3dasm.datageneration.functions.pybenchfunction.ThreeHump`
Trid                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Trid`
Xin She Yang N.3         :class:`~f3dasm.datageneration.functions.pybenchfunction.XinSheYangN3`
======================== ====================================================================================

Multimodal functions
^^^^^^^^^^^^^^^^^^^^

======================== ====================================================================================
Name                      Docs of the Python class                                              
======================== ====================================================================================
Ackley                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Ackley`
Ackley N. 3              :class:`~f3dasm.datageneration.functions.pybenchfunction.AckleyN3`
Ackley N. 4              :class:`~f3dasm.datageneration.functions.pybenchfunction.AckleyN4`
Adjiman                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Adjiman`
Bartels                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Bartels`
Beale                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Beale`
Bird                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Bird`
Bohachevsky N. 2         :class:`~f3dasm.datageneration.functions.pybenchfunction.BohachevskyN2`
Bohachevsky N. 3         :class:`~f3dasm.datageneration.functions.pybenchfunction.BohachevskyN3`
Branin                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Branin`
Bukin N. 6               :class:`~f3dasm.datageneration.functions.pybenchfunction.BukinN6`
Colville                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Colville`
Cross-in-Tray            :class:`~f3dasm.datageneration.functions.pybenchfunction.CrossInTray`
De Jong N. 5             :class:`~f3dasm.datageneration.functions.pybenchfunction.DeJongN5`
Deckkers-Aarts           :class:`~f3dasm.datageneration.functions.pybenchfunction.DeckkersAarts`
Easom                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Easom`
Egg Crate                :class:`~f3dasm.datageneration.functions.pybenchfunction.EggCrate`
Egg Holder               :class:`~f3dasm.datageneration.functions.pybenchfunction.EggHolder`
Goldstein-Price          :class:`~f3dasm.datageneration.functions.pybenchfunction.GoldsteinPrice`
Happy Cat                :class:`~f3dasm.datageneration.functions.pybenchfunction.HappyCat`
Himmelblau               :class:`~f3dasm.datageneration.functions.pybenchfunction.Himmelblau`
Holder-Table             :class:`~f3dasm.datageneration.functions.pybenchfunction.HolderTable`
Keane                    :class:`~f3dasm.datageneration.functions.pybenchfunction.Keane`
Langermann               :class:`~f3dasm.datageneration.functions.pybenchfunction.Langermann`
Levy                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Levy`
Levy N. 13               :class:`~f3dasm.datageneration.functions.pybenchfunction.LevyN13`
McCormick                :class:`~f3dasm.datageneration.functions.pybenchfunction.McCormick`
Michalewicz              :class:`~f3dasm.datageneration.functions.pybenchfunction.Michalewicz`
Periodic                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Periodic`
Perm d, beta             :class:`~f3dasm.datageneration.functions.pybenchfunction.PermDBeta`
Qing                     :class:`~f3dasm.datageneration.functions.pybenchfunction.Qing`
Quartic                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Quartic`
Rastrigin                :class:`~f3dasm.datageneration.functions.pybenchfunction.Rastrigin`
Rosenbrock               :class:`~f3dasm.datageneration.functions.pybenchfunction.Rosenbrock`
Salomon                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Salomon`
Schwefel                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Schwefel`
Shekel                   :class:`~f3dasm.datageneration.functions.pybenchfunction.Shekel`
Shubert                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Shubert`
Shubert N. 3             :class:`~f3dasm.datageneration.functions.pybenchfunction.ShubertN3`
Shubert N. 4             :class:`~f3dasm.datageneration.functions.pybenchfunction.ShubertN4`
Styblinski Tank          :class:`~f3dasm.datageneration.functions.pybenchfunction.StyblinskiTank`
Thevenot                 :class:`~f3dasm.datageneration.functions.pybenchfunction.Thevenot`
Xin She Yang             :class:`~f3dasm.datageneration.functions.pybenchfunction.XinSheYang`
Xin She Yang N.2         :class:`~f3dasm.datageneration.functions.pybenchfunction.XinSheYangN2`
======================== ====================================================================================


Functions including a randomized term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ====================================================================================
Name                      Docs of the Python class                                              
======================== ====================================================================================
Quartic                  :class:`~f3dasm.datageneration.functions.pybenchfunction.Quartic`
Xin She Yang             :class:`~f3dasm.datageneration.functions.pybenchfunction.XinSheYang`
======================== ====================================================================================