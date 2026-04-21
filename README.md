# DyMoTree: Dynamic Cell Fate Modeling Based on Tree-Structured Neural Network

DyMoTree is a integrated computaional framework for modeling dynamic cell fate transitions by integrating lineage tree structures with single-cell transcriptomic data. It enables robust inference of cell fate bias, identification of fate-specific states, and discovery of underlying molecular mechanisms.

---

## 1. Lineage graph construction

![Cell development landscape](./images/model.png)

**Description**:  
This graph represents the conceptual landscape of cell differentiation, inspired by Waddington’s epigenetic landscape. It illustrates how progenitor cells evolve into distinct terminal cell types through branching developmental trajectories.

---

## 2. DyMoTree Framework

![Network structure](./images/framework.png)

---

## Requirements

To run DyMoTree, ensure the following dependencies are installed:

* **Core Computing**: `numpy==1.26.4`, `pandas==2.3.2`, `scipy==1.11.4`, `jax==0.6.1`, `jaxlib==0.6.1`.
* **Deep Learning & GNN**: `torch==2.7.1`, `torch-geometric==2.6.1`.
* **Single-Cell Analysis**: `scanpy==1.11.4`.
* **Visualization**: `matplotlib==3.10.6`, `seaborn==0.13.2`.

Install them using pip:
```
pip install -r requirements.txt
```

## Quick Start

### 1. Command Line Usage

You can run DyMoTree experiments directly using the provided script with a YAML configuration file:

```
python run/run_dymotree.py \
  --config config/Fig2.bench.lt.yaml \
  --output_csv results/fate_prediction_results.csv
```
### 2. Python API Example

DyMoTree can also be seamlessly integrated into your Python workflow:

```
from dmt import DyMoTree
import scanpy as sc

# 1. Load your data
adata = sc.read_h5ad("your_data.h5ad")

# 2. Initialize the model
dmt = DyMoTree(
    adata=adata,
    k=15,
    progenitor='HSPC',
    terminal=['Neutrophil', 'Monocyte'],
    lineage_col='lineage',
    emb_key='pca',
    seed=42,
    device='cuda'
)

# 3. Build the lineage graph
dmt.lineage_graph(mask_threshold=0.8, mode='composite')

# 4. Train the model
dmt.train(iter={'formal': 300, 'intra': 100, 'lineage': 200})

# 5. Downstream analysis: identify driver genes
drivers = dmt.find_driver(progenitor='HSPC')
```

