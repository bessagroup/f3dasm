``f3dasm``: Framework for Data-Driven Design and Analysis of Structures and Materials
=====================================================================================

*April 3rd, 2023* *Code Release Week #1*

Table of contents
=================

**1. Project introduction** - 1.1 Overview - 1.2 Installation - 1.3
Getting started

**2. Demonstration**

1.1 Overview
------------

``f3dasm`` is one python package that consists of 8 submodules:

**Use ``f3dasm`` to handle your design of experiments**

Modules: - ``f3dasm.design`` - ``f3dasm.experiment``

**Use ``f3dasm`` to compare models**

Modules: - ``f3dasm.machinelearning`` - ``f3dasm.optimization`` -
``f3dasm.sampling``

**Use ``f3dasm`` to generate data**

Modules: - ``f3dasm.functions`` - ``f3dasm.data`` -
``f3dasm.simulation``


2. Demonstration
----------------

In the past practical sessions, I have shown you how to use ``f3dasm``
to benchmark various parts of a data-driven machine learning process
(**use-case #2**).

Today I will show you how to use ``f3dasm`` to streamline your own
data-driven process (**use-case #1**)

Import some other packages and set a seed

.. code:: ipython3

    import numpy as np
    import logging
    from pathos.helpers import mp  # For multiprocessing!
    import time # For ... timing!
    
    SEED = 42
    np.random.seed(SEED)

Example: Set up your program with f3dasm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say we have a program that we want to execute. It is important
that this could be **anything**. Like: - Calculate the loss of some
compliance curve in topology optimization! - Computing the mean stress
and strain from some abaqus simulation! - Benchmarking various
regressors in a multi-fidelity setting! - Create some parameter files
and call this cool CRATE program!

At the top level of your experiment, you will probably have a main
function that accepts some arguments and returns the quantity of
interest.

Let's create such a function, just for demonstration purposes.

.. code:: ipython3

    def main(a: float, b: float, c: float) -> float:    
        functions = [f3dasm.functions.Rastrigin, f3dasm.functions.Levy, f3dasm.functions.Ackley]
        y = []
        for func in functions:
            f = func(dimensionality=3, scale_bounds=np.tile([-1.,1.], (3,1)), seed=SEED)
            time.sleep(.1)
            y.append(f(np.array([a,b,c])).ravel()[0])
    
        # Sum the values
        out = sum(y)
        logging.info(f"Executed program with a={a:.3f}, b={b:.3f}, c={c:.3f}: \t Result {out:.3f}")
        return out

What are we seeing: - The program requires three floating points and
returns a float as well. - It creates three 3D-benchmark functions,
evaluates them sequentially and sums the results - We simulate some
computational cost (0.1 seconds per evaluation) by calling the
``time.sleep()`` method - We write to a log

   Note: ``my_own_program`` uses the integrated benchmark functions from
   ``f3dasm``, but this could very well be one of your codes without any
   dependency on ``f3dasm``.

Executing multiple experiments is easy:

.. code:: ipython3

    inputs = np.random.uniform(size=(10,3))
    
    start_time = time.perf_counter()
    outputs = np.array([main(*input_vals) for input_vals in inputs])
    time_not_parallel = time.perf_counter() - start_time
    
    print(f"It took {time_not_parallel:.5f} seconds to execute this for loop")


.. parsed-literal::

    2023-04-03 14:32:44,988 - Executed program with a=0.375, b=0.951, c=0.732: 	 Result 74.525
    2023-04-03 14:32:45,291 - Executed program with a=0.599, b=0.156, c=0.156: 	 Result 184.928
    2023-04-03 14:32:45,594 - Executed program with a=0.058, b=0.866, c=0.601: 	 Result 58.301
    2023-04-03 14:32:45,897 - Executed program with a=0.708, b=0.021, c=0.970: 	 Result 168.786
    2023-04-03 14:32:46,200 - Executed program with a=0.832, b=0.212, c=0.182: 	 Result 165.645
    2023-04-03 14:32:46,503 - Executed program with a=0.183, b=0.304, c=0.525: 	 Result 77.913
    2023-04-03 14:32:46,806 - Executed program with a=0.432, b=0.291, c=0.612: 	 Result 90.612
    2023-04-03 14:32:47,109 - Executed program with a=0.139, b=0.292, c=0.366: 	 Result 74.271
    2023-04-03 14:32:47,412 - Executed program with a=0.456, b=0.785, c=0.200: 	 Result 94.007
    2023-04-03 14:32:47,715 - Executed program with a=0.514, b=0.592, c=0.046: 	 Result 94.061


.. parsed-literal::

    It took 3.03013 seconds to execute this for loop


We can save the values of ``outputs`` for later use

This process (``main.py``) can be described with the following figure:



Local parallelization
~~~~~~~~~~~~~~~~~~~~~

If you are familiar with
`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`__,
you might already know that we can speed-up this function by parellizing
the internal for loop:

We create a multiprocessing pool (``mp.Pool()``) where we map the
functions to cores in our machine:

.. code:: ipython3

    def main_parallel(a: float, b: float, c: float) -> float:
        def evaluate_function(func, a, b, c):
            f = func(dimensionality=3, scale_bounds=np.tile([-1.,1.], (3,1)))
            y = f(np.array([a,b,c])).ravel()[0]
            time.sleep(.1)
            return y
    
        functions = [f3dasm.functions.Rastrigin, f3dasm.functions.Levy, f3dasm.functions.Ackley]
        with mp.Pool() as pool:
            y = pool.starmap(evaluate_function, [(func, a, b, c) for func in functions])
    
        # Sum the values
        out = sum(y)
    
        logging.info(f"Executed program with a={a:.3f}, b={b:.3f}, c={c:.3f}: \t Result: {out:.3f}")
        return out

Executing this function will speed up the process

.. code:: ipython3

    inputs = np.random.uniform(size=(10,3))
    
    start_time = time.perf_counter()
    outputs = np.array([main_parallel(*input_vals) for input_vals in inputs])
    time_parallel = time.perf_counter() - start_time
    
    print(f"It took {time_parallel:.5f} seconds to execute this for loop")
    print(f"We are {time_not_parallel-time_parallel:.5f} seconds faster by parellelization!")


.. parsed-literal::

    2023-04-03 14:32:47,939 - Executed program with a=0.599, b=0.156, c=0.156: 	 Result: 125.903
    2023-04-03 14:32:48,138 - Executed program with a=0.058, b=0.866, c=0.601: 	 Result: 91.501
    2023-04-03 14:32:48,379 - Executed program with a=0.708, b=0.021, c=0.970: 	 Result: 77.984
    2023-04-03 14:32:48,588 - Executed program with a=0.832, b=0.212, c=0.182: 	 Result: 114.672
    2023-04-03 14:32:48,808 - Executed program with a=0.183, b=0.304, c=0.525: 	 Result: 138.112
    2023-04-03 14:32:49,027 - Executed program with a=0.432, b=0.291, c=0.612: 	 Result: 88.281
    2023-04-03 14:32:49,236 - Executed program with a=0.139, b=0.292, c=0.366: 	 Result: 129.742
    2023-04-03 14:32:49,494 - Executed program with a=0.456, b=0.785, c=0.200: 	 Result: 62.062
    2023-04-03 14:32:49,731 - Executed program with a=0.514, b=0.592, c=0.046: 	 Result: 67.328
    2023-04-03 14:32:49,939 - Executed program with a=0.608, b=0.171, c=0.065: 	 Result: 119.909


.. parsed-literal::

    It took 2.19749 seconds to execute this for loop
    We are 0.83264 seconds faster by parellelization!


This process (``main_parallel.py``) can be described with the following
figure:



Scale-up: challenges
~~~~~~~~~~~~~~~~~~~~

Now we would like to really scale things up.

Q) What challenges lie along the way?

I asked ChatGPT:

-  **1. Experiment design and analysis**: As the complexity of the
   experiment increases, it becomes more difficult to design experiments
   that are robust and reproducible, and to analyze the results in a
   meaningful way. This can lead to issues with experimental design,
   parameter tuning, and statistical analysis.

-  **2. Parallelization**: As experiments become larger, it may be
   necessary to parallelize or distribute the computations across
   multiple machines or nodes in order to reduce the overall runtime.
   This introduces additional challenges such as synchronization between
   distributed processes.

-  **3. Managing data**: As the volume of data generated by an
   experiment increases, it becomes more difficult to manage and store
   that data. This can lead to issues with data corruption, loss, or
   inconsistency.

This is where ``f3dasm`` is a helping hand!

1. Experiment design and analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can create a ``f3dasm.DesignSpace`` to capture the variables of
interest: - A ``f3dasm.DesignSpace`` consists of an input and output
list of ``f3dasm.Parameter`` objects

.. code:: ipython3

    param_a = f3dasm.ContinuousParameter(name='a', lower_bound=-1., upper_bound=1.)
    param_b = f3dasm.ContinuousParameter(name='b', lower_bound=-1., upper_bound=1.)
    param_c = f3dasm.ContinuousParameter(name='c', lower_bound=-1., upper_bound=1.)
    param_out = f3dasm.ContinuousParameter(name='y')
    
    design = f3dasm.DesignSpace(input_space=[param_a, param_b, param_c], output_space=[param_out])

We can create an object to store the experiments:
``f3dasm.ExperimentData``, but we can also **sample from this
designspace** We do that with the ``f3dasm.sampling`` submodule:

   Note that this submodule offers an extension (``f3dasm[sampling]``)
   that include sampling strategies from ``SALib``

.. code:: ipython3

    # Create the sampler object
    sampler = f3dasm.sampling.RandomUniform(design=design, seed=SEED)
    
    data: f3dasm.ExperimentData = sampler.get_samples(numsamples=10)

The data object is under the hood a pandas dataframe:

.. code:: ipython3

    data.data




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th colspan="3" halign="left">input</th>
          <th>output</th>
        </tr>
        <tr>
          <th></th>
          <th>a</th>
          <th>b</th>
          <th>c</th>
          <th>y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.250920</td>
          <td>0.901429</td>
          <td>0.463988</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.197317</td>
          <td>-0.687963</td>
          <td>-0.688011</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.883833</td>
          <td>0.732352</td>
          <td>0.202230</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.416145</td>
          <td>-0.958831</td>
          <td>0.939820</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.664885</td>
          <td>-0.575322</td>
          <td>-0.636350</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>5</th>
          <td>-0.633191</td>
          <td>-0.391516</td>
          <td>0.049513</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>6</th>
          <td>-0.136110</td>
          <td>-0.417542</td>
          <td>0.223706</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>7</th>
          <td>-0.721012</td>
          <td>-0.415711</td>
          <td>-0.267276</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>8</th>
          <td>-0.087860</td>
          <td>0.570352</td>
          <td>-0.600652</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.028469</td>
          <td>0.184829</td>
          <td>-0.907099</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



The ``y`` values are NaN because we haven’t evaluate our experiment yet!
Let’s do that:

Handy: we can retrieve the input columns of a specific row as a
dictionary

.. code:: ipython3

    data.get_inputdata_by_index(index=3)




.. parsed-literal::

    {'a': 0.416145155592091, 'b': -0.9588310114083951, 'c': 0.9398197043239886}



Unpacking the values as arguments of our experiment creates the same
results:

.. code:: ipython3

    for index in range(data.get_number_of_datapoints()):
        value = main_parallel(**data.get_inputdata_by_index(index))
        data.set_outputdata_by_index(index, value)


.. parsed-literal::

    2023-04-03 14:32:50,297 - Executed program with a=-0.251, b=0.901, c=0.464: 	 Result: 261.134
    2023-04-03 14:32:50,539 - Executed program with a=0.197, b=-0.688, c=-0.688: 	 Result: 19.109
    2023-04-03 14:32:50,784 - Executed program with a=-0.884, b=0.732, c=0.202: 	 Result: 321.825
    2023-04-03 14:32:51,018 - Executed program with a=0.416, b=-0.959, c=0.940: 	 Result: 170.930
    2023-04-03 14:32:51,275 - Executed program with a=0.665, b=-0.575, c=-0.636: 	 Result: 79.458
    2023-04-03 14:32:51,527 - Executed program with a=-0.633, b=-0.392, c=0.050: 	 Result: 139.412
    2023-04-03 14:32:51,770 - Executed program with a=-0.136, b=-0.418, c=0.224: 	 Result: 115.536
    2023-04-03 14:32:52,015 - Executed program with a=-0.721, b=-0.416, c=-0.267: 	 Result: 83.109
    2023-04-03 14:32:52,253 - Executed program with a=-0.088, b=0.570, c=-0.601: 	 Result: 215.214
    2023-04-03 14:32:52,512 - Executed program with a=0.028, b=0.185, c=-0.907: 	 Result: 109.803


Now our data-object is filled

.. code:: ipython3

    data.data




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th colspan="3" halign="left">input</th>
          <th>output</th>
        </tr>
        <tr>
          <th></th>
          <th>a</th>
          <th>b</th>
          <th>c</th>
          <th>y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.250920</td>
          <td>0.901429</td>
          <td>0.463988</td>
          <td>261.134214</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.197317</td>
          <td>-0.687963</td>
          <td>-0.688011</td>
          <td>19.109039</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.883833</td>
          <td>0.732352</td>
          <td>0.202230</td>
          <td>321.825051</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.416145</td>
          <td>-0.958831</td>
          <td>0.939820</td>
          <td>170.930424</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.664885</td>
          <td>-0.575322</td>
          <td>-0.636350</td>
          <td>79.458296</td>
        </tr>
        <tr>
          <th>5</th>
          <td>-0.633191</td>
          <td>-0.391516</td>
          <td>0.049513</td>
          <td>139.411721</td>
        </tr>
        <tr>
          <th>6</th>
          <td>-0.136110</td>
          <td>-0.417542</td>
          <td>0.223706</td>
          <td>115.535908</td>
        </tr>
        <tr>
          <th>7</th>
          <td>-0.721012</td>
          <td>-0.415711</td>
          <td>-0.267276</td>
          <td>83.109400</td>
        </tr>
        <tr>
          <th>8</th>
          <td>-0.087860</td>
          <td>0.570352</td>
          <td>-0.600652</td>
          <td>215.214311</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.028469</td>
          <td>0.184829</td>
          <td>-0.907099</td>
          <td>109.803282</td>
        </tr>
      </tbody>
    </table>
    </div>



This process can be described with the following figure:



``f3dasm`` can handle the experiment distribution.

In order to set this up, navigate to a folder where you want to create
your experiment and run ``f3dasm.experiment.quickstart()``:

.. code:: ipython3

    # I'll not run this command because this is a demo
    
    # f3dasm.experiment.quickstart()

This creates the following files and folders:

::

   └── my_experiment 
       ├── main.py
       ├── config.py
       ├── config.yaml
       ├── default.yaml
       ├── pbsjob.sh
       └── README.md
       └── hydra/job_logging
           └── custom_script.py

Without going to much in detail, the following things have already been
set up automatically:

**Logging** - ``hydra`` (and the ``custom_script.py``) take care of all
(multiprocess) logging - including writing across nodes when executing
arrayjobs!

**Parameter storage** - ``config.yaml``, ``config.py`` and
``default.yaml`` can be used for easy reproducibility and parameter
tuning of your experiment!

**Parallelization** - ``pbsjob.sh`` can be used to execute your
``main.py`` file on the HPC, including array-jobs.

example:

::

   qsub pbsjob.sh
   qsub pbsjob.sh -t 0-10

**Saving data** - ``hydra`` creates a new ``outputs/<HPC JOBID>/``
directory that saves all output files, logs and settings when executing
``main.py`` - When executing arrayjobs, all arrayjobs write to the same
folder!

2. Parallelization
^^^^^^^^^^^^^^^^^^

Let’s recall: our single node process with ``f3dasm.ExperimentData`` can
be abstracted by the following image:



Parallelizing the **outer loop** is more difficult, but we can do that
across nodes with help of the ``f3dasm.experiment.JobQueue``

.. code:: ipython3

    job_queue = f3dasm.experiment.JobQueue(filename='my_jobs')


We can fill the queue with the rows of the ``f3dasm.ExperimentData``
object:

.. code:: ipython3

    job_queue.create_jobs_from_experimentdata(data)
    job_queue




.. parsed-literal::

    {0: 'open', 1: 'open', 2: 'open', 3: 'open', 4: 'open', 5: 'open', 6: 'open', 7: 'open', 8: 'open', 9: 'open'}



10 jobs have been added and they are all up for grabs!

Let’s first write this to disk so multiple nodes can access it:

.. code:: ipython3

    job_queue.write_new_jobfile()

A node can grab the first available job in the queue with the ``get()``
method: The file is locked when accessing the information from the JSON
file

.. code:: ipython3

    job_id = job_queue.get()
    print(f"The first open job_id is {job_id}!")


.. parsed-literal::

    The first open job_id is 0!


After returning the ``job_id``, the lock is removed and the job is
changed to ``in progress``

.. code:: ipython3

    job_queue.get_jobs()




.. parsed-literal::

    {0: 'in progress',
     1: 'open',
     2: 'open',
     3: 'open',
     4: 'open',
     5: 'open',
     6: 'open',
     7: 'open',
     8: 'open',
     9: 'open'}



When a new node asks a new job, it will return the next open job in
line!

.. code:: ipython3

    job_id = job_queue.get()
    print(f"The first open job_id is {job_id}!")


.. parsed-literal::

    The first open job_id is 1!


When a job is finished, you can mark it finished or with an error:

.. code:: ipython3

    job_queue.mark_finished(index=0)
    job_queue.mark_error(index=1)
    
    job_queue.get_jobs()




.. parsed-literal::

    {0: 'finished',
     1: 'error',
     2: 'open',
     3: 'open',
     4: 'open',
     5: 'open',
     6: 'open',
     7: 'open',
     8: 'open',
     9: 'open'}



We can now change our simple script to handle multiprocessing across
nodes!

.. code:: ipython3

    job_queue = f3dasm.experiment.JobQueue(filename='my_jobs2')
    job_queue.create_jobs_from_experimentdata(data)
    
    job_queue.write_new_jobfile()
    
    data.store('data')
    
    while True:
        try:
            jobnumber = job_queue.get()
        except f3dasm.experiment.NoOpenJobsError:
            break
        
        data = f3dasm.design.load_experimentdata('data')
        args = data.get_inputdata_by_index(jobnumber)
    
        value = main_parallel(**args)
        data.set_outputdata_by_index(jobnumber, value)
    
        data.store('data')
    
        job_queue.mark_finished(jobnumber)
    
    data.data


.. parsed-literal::

    2023-04-03 14:32:52,879 - Executed program with a=-0.251, b=0.901, c=0.464: 	 Result: 261.134
    2023-04-03 14:32:53,162 - Executed program with a=0.197, b=-0.688, c=-0.688: 	 Result: 19.109
    2023-04-03 14:32:53,451 - Executed program with a=-0.884, b=0.732, c=0.202: 	 Result: 321.825
    2023-04-03 14:32:53,714 - Executed program with a=0.416, b=-0.959, c=0.940: 	 Result: 170.930
    2023-04-03 14:32:53,963 - Executed program with a=0.665, b=-0.575, c=-0.636: 	 Result: 79.458
    2023-04-03 14:32:54,209 - Executed program with a=-0.633, b=-0.392, c=0.050: 	 Result: 139.412
    2023-04-03 14:32:54,436 - Executed program with a=-0.136, b=-0.418, c=0.224: 	 Result: 115.536
    2023-04-03 14:32:54,707 - Executed program with a=-0.721, b=-0.416, c=-0.267: 	 Result: 83.109
    2023-04-03 14:32:54,962 - Executed program with a=-0.088, b=0.570, c=-0.601: 	 Result: 215.214
    2023-04-03 14:32:55,193 - Executed program with a=0.028, b=0.185, c=-0.907: 	 Result: 109.803
    2023-04-03 14:32:55,203 - An unexpected error occurred: The jobfile my_jobs2 does not have any open jobs left!




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th colspan="3" halign="left">input</th>
          <th>output</th>
        </tr>
        <tr>
          <th></th>
          <th>a</th>
          <th>b</th>
          <th>c</th>
          <th>y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.250920</td>
          <td>0.901429</td>
          <td>0.463988</td>
          <td>261.134214</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.197317</td>
          <td>-0.687963</td>
          <td>-0.688011</td>
          <td>19.109039</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.883833</td>
          <td>0.732352</td>
          <td>0.202230</td>
          <td>321.825051</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.416145</td>
          <td>-0.958831</td>
          <td>0.939820</td>
          <td>170.930424</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.664885</td>
          <td>-0.575322</td>
          <td>-0.636350</td>
          <td>79.458296</td>
        </tr>
        <tr>
          <th>5</th>
          <td>-0.633191</td>
          <td>-0.391516</td>
          <td>0.049513</td>
          <td>139.411721</td>
        </tr>
        <tr>
          <th>6</th>
          <td>-0.136110</td>
          <td>-0.417542</td>
          <td>0.223706</td>
          <td>115.535908</td>
        </tr>
        <tr>
          <th>7</th>
          <td>-0.721012</td>
          <td>-0.415711</td>
          <td>-0.267276</td>
          <td>83.109400</td>
        </tr>
        <tr>
          <th>8</th>
          <td>-0.087860</td>
          <td>0.570352</td>
          <td>-0.600652</td>
          <td>215.214311</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.028469</td>
          <td>0.184829</td>
          <td>-0.907099</td>
          <td>109.803282</td>
        </tr>
      </tbody>
    </table>
    </div>



This process looks like this:



3. Managing data
~~~~~~~~~~~~~~~~

Sometimes you don’t want to write directly to the ``ExperimentData``
file. Perhaps the output is not a simple set of values, or you want to
do some post-processing. This is where the ``f3dasm.Filehandler`` comes
in handy.



You can create your own custom ``FileHandler`` by inheriting from the
``f3dasm.experiment.Filenhandler`` class: Upon initializing, you have to
provide: 
- the directory to check for created files 
- the suffix extension (like ``.csv``) of the files 
- files following the above pattern that are intentionally ignored (optional)

.. code:: ipython3

    class MyFilehandler(f3dasm.experiment.FileHandler):
        def execute(self, filename: str) -> int:
            # Do some post processing with the created file
            ...
            # Return an errorcode: 0 = succesful, 1 = error

End of the demonstration!
-------------------------

*Thank you for listening :)*
