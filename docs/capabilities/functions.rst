Benchmark functions
===================

Usage
-----


Creating the function
^^^^^^^^^^^^^^^^^^^^^

First we make a 2-dimensional continous design space with the helper function :func:`~f3dasm.base.utils.make_nd_continuous_design`:

.. code-block:: python

  bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
  design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensions=2)

Then we create an object of the :class:`~f3dasm.functions.pybenchfunction.Sphere` class, specifying the :attr:`~f3dasm.base.function.Function.dimensionality`:

.. code-block:: python
 
  sphere_function = f3dasm.functions.Sphere(dimensionality=dim, scale_bounds=bounds)

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
  
The gradient is estimated using the numdifftools package.

Plotting
^^^^^^^^

We can plot a 3D or 2D representation of the loss landscape:

.. code-block:: python

  sphere_function.plot(orientation="3D", domain=bounds)
  
img


.. code-block:: python

  sphere_function.plot(orientation="2D", domain=bounds)
  
img

Implement your own benchmark functions
--------------------------------------


Implemented benchmark functions
-------------------------------

The following implementations of benchmark functions can found under the :mod:`f3dasm.functions` module.
These are taken and modified from the `Python Benchmark Test Optimization Function Single Objective <https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective>`_ github repository.

Convex functions
^^^^^^^^^^^^^^^^

======================== ======================================================================
Name                      Docs of the Python class                                              
======================== ======================================================================
Ackley N. 2              :class:`f3dasm.functions.pybenchfunction.AckleyN2`
Bohachevsky N. 1         :class:`f3dasm.functions.pybenchfunction.BohachevskyN1`
Booth                    :class:`f3dasm.functions.pybenchfunction.Booth`
Brent                    :class:`f3dasm.functions.pybenchfunction.Brent`
Brown                    :class:`f3dasm.functions.pybenchfunction.Brown`
Bukin N. 6               :class:`f3dasm.functions.pybenchfunction.BukinN6`
Dixon Price              :class:`f3dasm.functions.pybenchfunction.DixonPrice`
Exponential              :class:`f3dasm.functions.pybenchfunction.Exponential`
Matyas                   :class:`f3dasm.functions.pybenchfunction.Matyas`
McCormick                :class:`f3dasm.functions.pybenchfunction.McCormick`
Perm 0, d, beta          :class:`f3dasm.functions.pybenchfunction.PermZeroDBeta`
Powell                   :class:`f3dasm.functions.pybenchfunction.Powell`
Rotated Hyper-Ellipsoid  :class:`f3dasm.functions.pybenchfunction.RotatedHyperEllipsoid`
Schwefel 2.20            :class:`f3dasm.functions.pybenchfunction.Schwefel2_20`
Schwefel 2.21            :class:`f3dasm.functions.pybenchfunction.Schwefel2_21`
Schwefel 2.22            :class:`f3dasm.functions.pybenchfunction.Schwefel2_22`
Schwefel 2.23            :class:`f3dasm.functions.pybenchfunction.Schwefel2_23`
Sphere                   :class:`f3dasm.functions.pybenchfunction.Sphere`
Sum Squares              :class:`f3dasm.functions.pybenchfunction.SumSquares`
Thevenot                 :class:`f3dasm.functions.pybenchfunction.Thevenot`
Trid                     :class:`f3dasm.functions.pybenchfunction.Trid`
Xin She Yang N.3         :class:`f3dasm.functions.pybenchfunction.XinSheYangN3`
Xin-She Yang N.4         :class:`f3dasm.functions.pybenchfunction.XinSheYangN4`
======================== ======================================================================


Seperable functions
^^^^^^^^^^^^^^^^^^^

======================== ======================================================================
Name                      Docs of the Python class                                              
======================== ======================================================================
Ackley                   :class:`f3dasm.functions.pybenchfunction.Ackley`
Bohachevsky N. 1         :class:`f3dasm.functions.pybenchfunction.BohachevskyN1`
Easom                    :class:`f3dasm.functions.pybenchfunction.Easom`
Egg Crate                :class:`f3dasm.functions.pybenchfunction.EggCrate`
Exponential              :class:`f3dasm.functions.pybenchfunction.Exponential`
Griewank                 :class:`f3dasm.functions.pybenchfunction.Griewank`
Michalewicz              :class:`f3dasm.functions.pybenchfunction.Michalewicz`
Powell                   :class:`f3dasm.functions.pybenchfunction.Powell`
Qing                     :class:`f3dasm.functions.pybenchfunction.Qing`
Quartic                  :class:`f3dasm.functions.pybenchfunction.Quartic`
Rastrigin                :class:`f3dasm.functions.pybenchfunction.Rastrigin`
Schwefel                 :class:`f3dasm.functions.pybenchfunction.Schwefel`
Schwefel 2.20            :class:`f3dasm.functions.pybenchfunction.Schwefel2_20`
Schwefel 2.21            :class:`f3dasm.functions.pybenchfunction.Schwefel2_21`
Schwefel 2.22            :class:`f3dasm.functions.pybenchfunction.Schwefel2_22`
Schwefel 2.23            :class:`f3dasm.functions.pybenchfunction.Schwefel2_23`
Sphere                   :class:`f3dasm.functions.pybenchfunction.Sphere`
Styblinski Tank          :class:`f3dasm.functions.pybenchfunction.StyblinskiTank`
Sum Squares              :class:`f3dasm.functions.pybenchfunction.SumSquares`
Thevenot                 :class:`f3dasm.functions.pybenchfunction.Thevenot`
Xin She Yang             :class:`f3dasm.functions.pybenchfunction.XinSheYang`
======================== ======================================================================


Differentiable functions
^^^^^^^^^^^^^^^^^^^^^^^^

======================== ======================================================================
Name                      Docs of the Python class                                              
======================== ======================================================================
Ackley                   :class:`f3dasm.functions.pybenchfunction.Ackley`
Ackley N. 2              :class:`f3dasm.functions.pybenchfunction.AckleyN2`
Ackley N. 3              :class:`f3dasm.functions.pybenchfunction.AckleyN3`
Ackley N. 4              :class:`f3dasm.functions.pybenchfunction.AckleyN4`
Adjiman                  :class:`f3dasm.functions.pybenchfunction.Adjiman`
Beale                    :class:`f3dasm.functions.pybenchfunction.Beale`
Bird                     :class:`f3dasm.functions.pybenchfunction.Bird`
Bohachevsky N. 1         :class:`f3dasm.functions.pybenchfunction.BohachevskyN1`
Bohachevsky N. 2         :class:`f3dasm.functions.pybenchfunction.BohachevskyN2`
Bohachevsky N. 3         :class:`f3dasm.functions.pybenchfunction.BohachevskyN3`
Booth                    :class:`f3dasm.functions.pybenchfunction.Booth`
Branin                   :class:`f3dasm.functions.pybenchfunction.Branin`
Brent                    :class:`f3dasm.functions.pybenchfunction.Brent`
Brown                    :class:`f3dasm.functions.pybenchfunction.Brown`
Colville                 :class:`f3dasm.functions.pybenchfunction.Colville`
De Jong N. 5             :class:`f3dasm.functions.pybenchfunction.DeJongN5`
Deckkers-Aarts           :class:`f3dasm.functions.pybenchfunction.DeckkersAarts`
Dixon Price              :class:`f3dasm.functions.pybenchfunction.DixonPrice`
Drop-Wave                :class:`f3dasm.functions.pybenchfunction.DropWave`
Easom                    :class:`f3dasm.functions.pybenchfunction.Easom`
Egg Crate                :class:`f3dasm.functions.pybenchfunction.EggCrate`
Egg Holder               :class:`f3dasm.functions.pybenchfunction.EggHolder`
Exponential              :class:`f3dasm.functions.pybenchfunction.Exponential`
Goldstein-Price          :class:`f3dasm.functions.pybenchfunction.GoldsteinPrice`
Griewank                 :class:`f3dasm.functions.pybenchfunction.Griewank`
Happy Cat                :class:`f3dasm.functions.pybenchfunction.HappyCat`
Himmelblau               :class:`f3dasm.functions.pybenchfunction.Himmelblau`
Keane                    :class:`f3dasm.functions.pybenchfunction.Keane`
Langermann               :class:`f3dasm.functions.pybenchfunction.Langermann`
Leon                     :class:`f3dasm.functions.pybenchfunction.Leon`
Levy                     :class:`f3dasm.functions.pybenchfunction.Levy`
Levy N. 13               :class:`f3dasm.functions.pybenchfunction.LevyN13`
Matyas                   :class:`f3dasm.functions.pybenchfunction.Matyas`
McCormick                :class:`f3dasm.functions.pybenchfunction.McCormick`
Michalewicz              :class:`f3dasm.functions.pybenchfunction.Michalewicz`
Periodic                 :class:`f3dasm.functions.pybenchfunction.Periodic`
Perm d, beta             :class:`f3dasm.functions.pybenchfunction.PermDBeta`
Perm 0, d, beta          :class:`f3dasm.functions.pybenchfunction.PermZeroDBeta`
Qing                     :class:`f3dasm.functions.pybenchfunction.Qing`
Quartic                  :class:`f3dasm.functions.pybenchfunction.Quartic`
Rastrigin                :class:`f3dasm.functions.pybenchfunction.Rastrigin`
Ridge                    :class:`f3dasm.functions.pybenchfunction.Ridge`
Rosenbrock               :class:`f3dasm.functions.pybenchfunction.Rosenbrock`
Rotated Hyper-Ellipsoid  :class:`f3dasm.functions.pybenchfunction.RotatedHyperEllipsoid`
Salomon                  :class:`f3dasm.functions.pybenchfunction.Salomon`
Schaffel N. 1            :class:`f3dasm.functions.pybenchfunction.SchaffelN1`
Schaffel N. 2            :class:`f3dasm.functions.pybenchfunction.SchaffelN2`
Schaffel N. 3            :class:`f3dasm.functions.pybenchfunction.SchaffelN3`
Schaffel N. 4            :class:`f3dasm.functions.pybenchfunction.SchaffelN4`
Shekel                   :class:`f3dasm.functions.pybenchfunction.Shekel`
Shubert                  :class:`f3dasm.functions.pybenchfunction.Shubert`
Shubert N. 3             :class:`f3dasm.functions.pybenchfunction.ShubertN3`
Shubert N. 4             :class:`f3dasm.functions.pybenchfunction.ShubertN4`
Styblinski Tank          :class:`f3dasm.functions.pybenchfunction.StyblinskiTank`
Sum Squares              :class:`f3dasm.functions.pybenchfunction.SumSquares`
Thevenot                 :class:`f3dasm.functions.pybenchfunction.Thevenot`
Three-Hump               :class:`f3dasm.functions.pybenchfunction.ThreeHump`
Trid                     :class:`f3dasm.functions.pybenchfunction.Trid`
Xin She Yang N.3         :class:`f3dasm.functions.pybenchfunction.XinSheYangN3`
======================== ======================================================================

Multimodal functions
^^^^^^^^^^^^^^^^^^^^

======================== ======================================================================
Name                      Docs of the Python class                                              
======================== ======================================================================
Ackley                   :class:`f3dasm.functions.pybenchfunction.Ackley`
Ackley N. 3              :class:`f3dasm.functions.pybenchfunction.AckleyN3`
Ackley N. 4              :class:`f3dasm.functions.pybenchfunction.AckleyN4`
Adjiman                  :class:`f3dasm.functions.pybenchfunction.Adjiman`
Bartels                  :class:`f3dasm.functions.pybenchfunction.Bartels`
Beale                    :class:`f3dasm.functions.pybenchfunction.Beale`
Bird                     :class:`f3dasm.functions.pybenchfunction.Bird`
Bohachevsky N. 2         :class:`f3dasm.functions.pybenchfunction.BohachevskyN2`
Bohachevsky N. 3         :class:`f3dasm.functions.pybenchfunction.BohachevskyN3`
Branin                   :class:`f3dasm.functions.pybenchfunction.Branin`
Bukin N. 6               :class:`f3dasm.functions.pybenchfunction.BukinN6`
Colville                 :class:`f3dasm.functions.pybenchfunction.Colville`
Cross-in-Tray            :class:`f3dasm.functions.pybenchfunction.CrossInTray`
De Jong N. 5             :class:`f3dasm.functions.pybenchfunction.DeJongN5`
Deckkers-Aarts           :class:`f3dasm.functions.pybenchfunction.DeckkersAarts`
Easom                    :class:`f3dasm.functions.pybenchfunction.Easom`
Egg Crate                :class:`f3dasm.functions.pybenchfunction.EggCrate`
Egg Holder               :class:`f3dasm.functions.pybenchfunction.EggHolder`
Goldstein-Price          :class:`f3dasm.functions.pybenchfunction.GoldsteinPrice`
Happy Cat                :class:`f3dasm.functions.pybenchfunction.HappyCat`
Himmelblau               :class:`f3dasm.functions.pybenchfunction.Himmelblau`
Holder-Table             :class:`f3dasm.functions.pybenchfunction.HolderTable`
Keane                    :class:`f3dasm.functions.pybenchfunction.Keane`
Langermann               :class:`f3dasm.functions.pybenchfunction.Langermann`
Levy                     :class:`f3dasm.functions.pybenchfunction.Levy`
Levy N. 13               :class:`f3dasm.functions.pybenchfunction.LevyN13`
McCormick                :class:`f3dasm.functions.pybenchfunction.McCormick`
Michalewicz              :class:`f3dasm.functions.pybenchfunction.Michalewicz`
Periodic                 :class:`f3dasm.functions.pybenchfunction.Periodic`
Perm d, beta             :class:`f3dasm.functions.pybenchfunction.PermDBeta`
Qing                     :class:`f3dasm.functions.pybenchfunction.Qing`
Quartic                  :class:`f3dasm.functions.pybenchfunction.Quartic`
Rastrigin                :class:`f3dasm.functions.pybenchfunction.Rastrigin`
Rosenbrock               :class:`f3dasm.functions.pybenchfunction.Rosenbrock`
Salomon                  :class:`f3dasm.functions.pybenchfunction.Salomon`
Schwefel                 :class:`f3dasm.functions.pybenchfunction.Schwefel`
Shekel                   :class:`f3dasm.functions.pybenchfunction.Shekel`
Shubert                  :class:`f3dasm.functions.pybenchfunction.Shubert`
Shubert N. 3             :class:`f3dasm.functions.pybenchfunction.ShubertN3`
Shubert N. 4             :class:`f3dasm.functions.pybenchfunction.ShubertN4`
Styblinski Tank          :class:`f3dasm.functions.pybenchfunction.StyblinskiTank`
Thevenot                 :class:`f3dasm.functions.pybenchfunction.Thevenot`
Xin She Yang             :class:`f3dasm.functions.pybenchfunction.XinSheYang`
Xin She Yang N.2         :class:`f3dasm.functions.pybenchfunction.XinSheYangN2`
======================== ======================================================================


Functions including a randomized term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ======================================================================
Name                      Docs of the Python class                                              
======================== ======================================================================
Quartic                  :class:`f3dasm.functions.pybenchfunction.Quartic`
Xin She Yang             :class:`f3dasm.functions.pybenchfunction.XinSheYang`
======================== ======================================================================
