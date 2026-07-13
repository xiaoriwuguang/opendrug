<div align="center">
  <img src="LOGO.jpg" alt="OpenDrug Logo" border="0" width="100%"/>
</div>

<p align="center">
  <a href="https://opendrug.readthedocs.io/en/latest/">Docs</a> •
  <a href="#quick-start">Quick Start</a> •
</p>

# OpenDrug

## <span id="quick-start">🚀 Quick Start</span>

Follow these steps to get started with OpenDrug:

### **Step 1: Clone the Repository**

```
git clone <REPO-URL>
```

### **Step 2: Install Dependencies**

#### General Dependencies

You can install the general dependencies:

```
conda env create -f opendrug.yml
```

### **Step 3: Run the Main Script**

Train a DDI baseline:

```bash
python -m opendrug.main --model MRCGNN --matrix zhangddi
```

Train DrugBAN on BIOSNAP:

```bash
python -m opendrug.main --task dti --model DrugBAN --matrix BIOSNAP \
    --modality drug_smiles --protein_sequence protein_sequence \
    --ban_heads 4
```

See the full [Quickstart Example](https://opendrug.readthedocs.io/en/latest/) for more.
