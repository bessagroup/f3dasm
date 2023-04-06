JobQueue
========

Parallelizing the outer loop of a program is more difficult, but we can do that
across nodes with help of the :class:`~f3dasm.experiment.jobs.JobQueue`

.. code:: ipython3

    job_queue = f3dasm.experiment.JobQueue(filename='my_jobs')


We can fill the queue with the rows of the :class:`~f3dasm.design.experimentdata.ExperimentData`
object:

.. code:: ipython3

    job_queue.create_jobs_from_experimentdata(data)
    job_queue




.. parsed-literal::

    {0: 'open', 1: 'open', 2: 'open', 3: 'open', 4: 'open', 5: 'open', 6: 'open', 7: 'open', 8: 'open', 9: 'open'}



10 jobs have been added and they are all up for grabs!

Let's first write this to disk so multiple nodes can access it:

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