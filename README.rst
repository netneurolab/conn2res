**Development Status:** 3 - Alpha. Some features still need to be added and tested.

conn2res
========

``conn2res`` is a reservoir computing toolbox designed for neuroscientists 
to train connectome-informed reservoirs to perform cognitive tasks. The main 
advantages of the toolbox are its flexibility in terms of the connectivity matrix 
used for the reservoir, the local dynamics of the nodes and the possibility to 
select the input and output nodes, as well as a comprehensive corpus of 
neuroscience tasks -designed for supervised learning- provided by 
`NeuroGym <https://github.com/neurogym/neurogym>`_.

The accompanying manuscript has been uploaded to 
`bioRxiv <https://www.biorxiv.org/content/10.1101/2023.05.31.543092v1>`_.


Brief primer on Reservoir Computing
-----------------------------------

Reservoir computing is a computational paradigm that essentially exploits the rich 
dynamics of complex dynamical systems, such as artificial recurrent neural networks 
(RNNs), to compute with time-varying input data (Lukoševičius, M. and Jaeger, H, 2009). 
The conventional reservoir computing architecture consists of an input layer, followed 
by the reservoir and a readout module. Typically, the reservoir is a randomly 
connected RNN and the readout module a linear model. In contrast to traditional RNNs, 
the connections of the reservoir are fixed; only the weights that connect the 
reservoir to the readout module are trained, which correspond to the parameters 
of the linear model. These weights are trained in a supervised manner to learn the 
representations of the external stimuli constructed by the reservoir, and can be 
adapted to a wide range of tasks, including speech recognition, motor learning, 
natural language processing, working memory and spatial navigation. Because 
arbitrary network architecture and dynamics can be superimposed on the reservoir, 
implementing biologically plausible network architectures allows to investigate 
how brain network organization and dynamics jointly support learning. 

.. image:: source/images/rc.png
    :width: 600

conn2res: an overview
---------------------

The conn2res toolbox provides a general use-case driven workflow that takes as
input (1) either the type of task to be performed (see `NeuroGym
<https://github.com/neurogym/neurogym>`__), or a supervised dataset of input-
label pairs can also be provided; (2) a binary or weighted connectome, which
serves as the reservoir’s architecture; (3) the input nodes (i.e., nodes that
receive the external signal); (4) the readout nodes (i.e., nodes from which
information will be read and used to train the linear model); and (5) the type
of dynamics governing the activation of the reservoir’s units (continuous or
discrete time nonlinear dynamics can be implemented, including spiking neurons
or artificial neurons with different activation functions such as ReLU, leaky
ReLU, sigmoid or hyperbolic tangent). Depending on the type of dynamics, the
output is either a performance score, or a performance curve as a function of
the parameter that controls for the qualitative behavior of the reservoir’s
dynamics (i.e., stable, critical or chaotic).

.. image:: source/images/conn2res.png
    :width: 600

The toolbox has been extended to simulate physical connectome-informed
memristive reservoirs, a newly type of neuromorphic hardware that, thanks to
its high computational and energy efficiency, has the potential to replace
conventional computer chips and revolutionize artificial intelligence algorithms
(Tanaka, G., et al., 2019).


Installation requirements
-------------------------

Currently, ``conn2res`` works with Python 3.8+ and requires a few
dependencies:

- bctpy (>=0.5)
- gym (==0.21.0)
- matplotlib (>=3.5)
- neurogym (==0.0.1)
- numpy (>=1.22)
- pandas (>=1.4)
- reservoirpy (==0.3.5)
- scipy (>=1.7)
- scikit-learn (>=1.1)
- seaborn (>=0.11)

You can get started by installing ``conn2res`` from the source repository
with:

.. code-block:: bash

    git clone https://github.com/netneurolab/conn2res
    cd conn2res
    pip install . -r requirements.txt  # this is to make sure that all requirements are installed
    cd ..
    git clone -b v0.0.1 https://github.com/neurogym/neurogym.git
    cd neurogym
    pip install -e .

You are ready to go!

Citation
--------

If you use the ``conn2res`` toolbox, please cite our 
`paper <https://www.biorxiv.org/content/10.1101/2023.05.31.543092v1>`_.

License information
-------------------

This work is licensed under a BSD 3-Clause "New" or "Revised" License.
The full license can be found in the
`LICENSE <https://github.com/netneurolab/conn2res/blob/documentation/LICENSE>`_ 
file in the ``conn2res`` distribution.