.. _installation_setup:

----------------------
Installation and setup
----------------------

.. _basic_installation:

Basic installation
==================

This package requires Python 3.8+. Assuming you have the correct version of
Python installed, you can install ``conn2res`` by opening a terminal and 
running the following:

.. code-block:: bash

    pip install conn2res

Alternatively, you can install the most up-to-date version from GitHub:

.. code-block:: bash

   git clone https://github.com/netneurolab/conn2res.git
   cd conn2res
   pip install .
   cd ..

.. _installation_requirements:

Requirements
============

In order to effectively use ``conn2res`` you must have the 
`NeuroGym <https://github.com/neurogym/neurogym>`__ repository installed and
accesible in your computer. Right now, the only compatible version of ``NeuroGym``
with ``conn2res`` is ``Neurogym-v0.0.1``. Large part of the functionalities of 
the ``conn2res`` toolbox rely on a few functions from the Neurogym repository.
This version currently works only if installed in editable mode.  
To install this particular version of Neurogym you can type:

.. code-block:: bash

    git clone -b v0.0.1 https://github.com/neurogym/neurogym.git
    cd neurogym
    pip install -e .

You are ready to go!