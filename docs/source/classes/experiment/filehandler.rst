Filehandler
===========

You can create your own custom filehandler by inheriting from the
:class:`~f3dasm.experiment.filehandler.FileHandler` class: Upon initializing, you have to
provide: 

- the directory to check for created files 
- the suffix extension (like ``.csv``) of the files 
- files following the above pattern that are intentionally ignored (optional)

.. code:: python

    class MyFilehandler(f3dasm.experiment.FileHandler):
        def execute(self, filename: str) -> int:
            # Do some post processing with the created file
            ...
            # Return an errorcode: 0 = succesful, 1 = error