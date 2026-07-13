:mod:`opendrug.models`
=======================

Per-task model classes.  All inherit from ``torch.nn.Module``.  For a
narrative description of each baseline see :doc:`/models/index`.

DDI models
----------

.. toctree::
   :maxdepth: 1
   :caption: DDI classes

   models/MRCGNN
   models/GOGNN
   models/MVA
   models/DeepDDI
   models/DDKG
   models/SumGNN
   models/PHGLDDI
   models/ExDDI
   models/CASTER
   models/MKGFENN

DTI / DTA / Cold-start models
----------------------------

.. toctree::
   :maxdepth: 1
   :caption: DTI / DTA / Cold-start classes

   models/DTA
   models/DTI
   models/KGE_NFM
   models/MGraphDTA
   models/MMD_DTA
   models/RSGCL_DTI
   models/GraphDTA
   models/EviDTI
   models/DTIAM
   models/DrugBAN
   models/ColdstartCPI
   models/AdaMBind

PPI models
----------

.. toctree::
   :maxdepth: 1
   :caption: PPI classes

   models/DL_PPI
   models/TAGPPI
   models/PPI_TUnA
   models/MARPPI
   models/MAPE_PPI
   models/HIGH_PPI
   models/GTB_PPI
   models/GraphPPIS
   models/D_SCRIPT
   models/CollaPPI

PPI base class
--------------

.. automodule:: opendrug.models.PPI_model
   :members:
   :show-inheritance:
   :no-index:

Manager
-------

.. automodule:: opendrug.models.model_manager
   :members:
   :show-inheritance:
   :no-index:
