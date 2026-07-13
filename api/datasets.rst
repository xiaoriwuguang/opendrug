:mod:`opendrug.data`
===========================

Dataset loaders under :file:`opendrug/data/`.  All dataset classes
inherit from :class:`opendrug.data.BaseDataset` and provide the unified
``__getitem__`` / ``modal_dims`` API.

Dataset classes
---------------

.. toctree::
   :maxdepth: 1
   :caption: Dataset classes

   datasets/Unified_dataset
   datasets/BaseDataset
   datasets/ColdstartCPI_dataset
   datasets/DL_PPI_dataset
   datasets/DTA_dataset
   datasets/DTI_dataset
   datasets/GoGNN_dataset
   datasets/MRCGNN_dataset
   datasets/MUFFIN_dataset
   datasets/MVA_dataset
   datasets/PPI_dataset
   datasets/TAGPPI_dataset
   datasets/TIGER_dataset
   datasets/ZeroDDI_dataset

Manager
-------

.. automodule:: opendrug.data.dataset_manager
   :members:
   :show-inheritance:
   :no-index:

Reading the source
------------------

Each class above is documented from its raw file under
``opendrug/data/<lowercased-name>.py``.  If a particular page is empty,
install OpenDrug's runtime dependencies (DGL, torch-geometric, RDKit) and
re-build the docs locally — see :doc:`/installation`.
