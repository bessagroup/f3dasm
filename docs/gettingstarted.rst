.. _installation-instructions:

===============
Getting Started
===============

There are different ways to install scikit-learn:

  * :ref:`Install the latest official release <install_official_release>`. This
    is the best approach for most users that want to use the :code:`f3dasm` package.

  * :ref:`Building the package from source
    <install_from_source>`. This is for users who wish to contribute to the
    project.



.. _install_official_release:

Installing the latest release
=============================

:code:`f3dasm` is compatible with Python version 3.8


----

.. This quickstart installation is a hack of the awesome
   https://spacy.io/usage/#quickstart page.
   See the original javascript implementation
   https://github.com/ines/quickstart. 
   I took the implementation of scikit-learn 
   (https://scikit-learn.org/stable/_sources/install.rst.txt)

.. raw:: html

  <div class="install">
       <strong>Operating System</strong>
          <input type="radio" name="os" id="quickstart-win" checked>
          <label for="quickstart-win">Windows</label>
          <input type="radio" name="os" id="quickstart-mac">
          <label for="quickstart-mac">macOS</label>
          <input type="radio" name="os" id="quickstart-lin">
          <label for="quickstart-lin">Linux</label><br />
       <strong>Packager</strong>
          <input type="radio" name="packager" id="quickstart-pip" checked>
          <label for="quickstart-pip">pip</label>
          <input type="radio" name="packager" id="quickstart-conda">
          <label for="quickstart-conda">conda</label><br />
       </span>

----

.. raw:: html

       <div>
         <span class="sk-expandable" data-packager="pip" data-os="windows">Install the 64bit version of Python 3.8, for instance from <a href="https://www.python.org/">https://www.python.org</a>.</span
         ><span class="sk-expandable" data-packager="pip" data-os="mac">Install Python 3.8 using <a href="https://brew.sh/">homebrew</a> (<code>brew install python</code>) or by manually installing the package from <a href="https://www.python.org">https://www.python.org</a>.</span
         ><span class="sk-expandable" data-packager="pip" data-os="linux">Install python3 and python3-pip using the package manager of the Linux Distribution.</span
         ><span class="sk-expandable" data-packager="conda"
            >Install conda using the <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/">Anaconda or miniconda</a>
             installers (no administrator permission required for any of those).</span>
       </div>

|
Then run:

.. raw:: html

        <div class="highlight-console notranslate"><div class="highlight"><pre><span></span
          ><span class="sk-expandable" data-packager="pip" data-os="mac"><span class="gp">$ </span>pip install -U f3dasm</span
          ><span class="sk-expandable" data-packager="pip" data-os="windows"><span class="gp">$ </span>pip install -U f3dasm</span
          ><span class="sk-expandable" data-packager="pip" data-os="linux"><span class="gp">$ </span>pip install -U f3dasm</span
          ><span class="sk-expandable" data-packager="conda"><span class="gp">$ </span>conda create -n f3dasm_env python=3.8</span
          ><span class="sk-expandable" data-packager="conda"><span class="gp">$ </span>conda activate f3dasm_env</span
          ><span class="sk-expandable" data-packager="conda"><span class="gp">$ </span>conda install pip</span
          ><span class="sk-expandable" data-packager="conda"><span class="gp">$ </span>pip install -U f3dasm</span
          ></pre></div></div>


In order to check your installation you can use

.. code-block:: console

  $ python -c "import f3dasm; f3dasm.show_versions()"
  >>> F3DASM:
  >>>    f3dasm: 0.2.92
  >>>    ...

This will show the installed version of :code:`f3dasm` and the versions of the dependencies.


.. _install_from_source:

Installing from source
======================