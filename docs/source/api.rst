.. _api_ref:

.. currentmodule:: conn2res

Reference API
=============

.. contents:: **List of modules**
   :local:

.. _ref_tasks:

:mod:`conn2res.tasks` - Task dataset fetchers
--------------------------------------------------------
.. automodule:: conn2res.tasks
      :no-members:
      :no-inherited-members:

.. currentmodule:: conn2res.tasks

.. autosummary::
   :template: class.rst
   :toctree: generated/

   conn2res.tasks.Task
   conn2res.tasks.NeuroGymTask
   conn2res.tasks.Reservoirpy
   conn2res.tasks.Conn2ResTask

.. _ref_connectivity:

:mod:`conn2res.connectivity` - Connectivity data handling
-----------------------------------------------
.. automodule:: conn2res.connectivity
   :no-members:
   :no-inherited-members:

.. currentmodule:: conn2res.connectivity

.. autosummary::
   :template: class.rst
   :toctree: generated/

   conn2res.connectivity.Conn

.. _ref_reservoir:

:mod:`conn2res.reservoir` - Reservoir objects
------------------------------------------
.. automodule:: conn2res.reservoir
   :no-members:
   :no-inherited-members:

.. currentmodule:: conn2res.reservoir

.. autosummary::
   :template: class.rst
   :toctree: generated/

   conn2res.reservoir.Reservoir
   conn2res.reservoir.EchoStateNetwork
   conn2res.reservoir.MemristiveReservoir
   conn2res.reservoir.MSSNetwork
   
.. _ref_readout:

:mod:`conn2res.readout` - Readout module
---------------------------------------------------
.. automodule:: conn2res.readout
   :no-members:
   :no-inherited-members:

.. currentmodule:: conn2res.readout

.. autosummary::
   :template: class.rst
   :toctree: generated/

   conn2res.readout.Readout
  
.. autosummary::
   :template: function.rst
   :toctree: generated/

   conn2res.readout.train_test_split
   conn2res.readout.select_model
   conn2res.readout.regressor
   conn2res.readout.classifier
   conn2res.readout.multioutput_regressor
   conn2res.readout.multioutput_classifier
   conn2res.readout.multiclass_classifier

.. _ref_performance:

:mod:`conn2res.performance` - Performance metrics
---------------------------------------
.. automodule:: conn2res.performance
   :no-members:
   :no-inherited-members:

.. currentmodule:: conn2res.performance

.. autosummary::
   :template: function.rst
   :toctree: generated/

   conn2res.performance.r2_score
   conn2res.performance.mean_squared_error
   conn2res.performance.root_mean_squared_error
   conn2res.performance.mean_absolute_error
   conn2res.performance.corrcoef
   conn2res.performance.accuracy_score
   conn2res.performance.balanced_accuracy_score
   conn2res.performance.f1_score
   conn2res.performance.precision_score
   conn2res.performance.recall_score

.. _ref_plotting:

:mod:`conn2res.plotting` - Plotting functions
---------------------------------------
.. automodule:: conn2res.plotting
   :no-members:
   :no-inherited-members:

.. currentmodule:: conn2res.plotting

.. autosummary::
   :template: function.rst
   :toctree: generated/

   conn2res.plotting.transform_data
   conn2res.plotting.plot_iodata
   conn2res.plotting.plot_reservoir_states
   conn2res.plotting.plot_diagnostics
   conn2res.plotting.plot_performance
   conn2res.plotting.plot_phase_space
