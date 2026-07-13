---
orphan: true
---

.. OpenDrug documentation master file, created on 2026-07-13.
   You can adapt this file completely to your liking.

.. title:: OpenDrug

.. meta::
   :description: OpenDrug — Unified Drug-Related Prediction Framework
   :keywords: drug-drug interaction, DDI, DTI, DTA, PPI, machine learning,
              graph neural network, multi-modal, protein interaction

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT

OpenDrug
========

**OpenDrug** is a unified, modular framework for drug-related graph-prediction
tasks.  It brings together 33+ baseline models under a single training pipeline,
covering four task families:

* **DDI** — drug-drug interaction prediction (binary / multi-class / multi-label / zero-shot)
* **DTI** — drug-target interaction prediction (binary classification)
* **DTA** — drug-target affinity prediction (regression)
* **Cold-start CPI** — cold-start drug-target prediction

The framework is designed so that switching between models, datasets, and
modalities requires only changing a few command-line flags — no source code
modifications.  Results are logged to ``results/{model}/`` with deterministic
reproducibility across seeds.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   guide/tasks
   guide/datasets
   guide/modalities
   guide/configuration

.. toctree::
   :maxdepth: 2
   :caption: In Depth

   guide/training
   guide/evaluation
   guide/results
   models/index
   tutorials/ddi_binary
   tutorials/dti_classification
   tutorials/dta_regression
   tutorials/coldstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/datasets
   api/models
   api/trainers
   api/pipeline
   api/utils
   api/evaluate

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   changelog
   references

.. toctree::
   :maxdepth: 2
   :caption: Appendices

   references

----

Quick Example
-------------

Train PHGLDDI (a DDI model) on the ZhangDDI dataset::

    python -m opendrug.main \
        --model PHGLDDI \
        --matrix zhangddi \
        --modality smiles sequence 3d mechanism text \
        --epochs 150 \
        --batch 512 \
        --lr 1e-3

Train DrugBAN for DTI classification on BIOSNAP::

    python -m opendrug.main \
        --task dti \
        --model DrugBAN \
        --matrix BIOSNAP \
        --modality drug_smiles \
        --protein_sequence protein_sequence \
        --epochs 100 \
        --ban_heads 4

See :doc:`quickstart` for a step-by-step walk-through and
:doc:`installation` for setup instructions.
