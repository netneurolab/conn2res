This folder contains several use-case examples of how the ``conn2res`` toolbox
can be applied to address specific biological questions. The processed
connectivity data is publicly available and can be downloaded
`HERE <https://zenodo.org/records/10205004>`_.

Tutorial 1: Main conn2res workflow
=======================================================================
This tutorial consists of a detailed step-by-step example to
illustrate the main ``conn2res`` workflow in action.


Example 1: Inferences on global network organization
=======================================================================
This example shows how global computational capacity relates to global
network topology. Specifically, we implement 5 group-consensus human
connectomes as reservoirs (echo-state networks) to perform a memory
capacity task. The performance of each empirical connectome is then
compared against the performance of a family of 500 rewired nulls.


Example 2: Anatomical inferences
=======================================================================
This example demonstrates how the toolbox can be used to make inferences
about regional heterogeneity or specificity for computational capacity.
Specifically, we implement the perceptual decision making task on a
single subject-level, connectome-informed reservoir. Cortical nodes
are stratified according to their affiliation with the canonical
intrinsic networks. Brain regions in the visual network are used as
input nodes. To quantify task performance, each intrinsic networks
is used separately as a readout module.


Example 3: Cross-species comparison
=======================================================================
This example shows how the toolbox can be applied to compare networks
across species. Specifically, we implement connectomes reconstructed
from four different organisms, namely fruit fly, mouse, rat and
macaque to perform a memory capacity task. We compare task
performance in each empirical connectome with a population of 500
rewired null networks.
