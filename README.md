# AGNO: Attention-based Graph Neural Operators for Discrete Structures

This repository contains the implementation used for the accepted paper:

**Attention-based graph neural operators for learning parametric response mappings in discrete structures**

DOI: https://doi.org/10.1016/j.engappai.2026.115232

AGNO is a graph-based neural operator for predicting parametric structural responses of discrete truss and frame systems. The model combines local graph neural operator layers with a global, permutation-invariant attention mechanism, so it can represent both topology-aware member interactions and nonlocal structural response behavior.

The repository is prepared as a public code release. The `docs/` folder contains selected figures and documentation assets.

## Method Overview

AGNO learns parametric response mappings on discrete structural graphs. A structure is represented as a graph `G = (V, E)`, where nodes correspond to joints or frame nodes and edges correspond to structural members. The model maps node-indexed input features and edge-indexed geometric features to node-indexed structural responses:

```text
(node features, edge features, graph connectivity) -> nodal responses
```

In the implemented graph datasets, node features contain normalized coordinates and applied load components. Edge features contain member geometry, including direction cosines and normalized member length. The targets are nodal displacements for truss problems and nodal displacement/rotation components for frame problems.

The AGNO architecture follows an encode-local-global-decode structure:

```text
h0 = encoder(x)
h_{l+1} = GlobalAttentionBlock(GNOBlock(h_l, edge_index, edge_attr))
y = decoder(h_L)
```

The local `GNOBlock` learns an edge-conditioned kernel operator. For each edge, a neural network maps `edge_attr` to a dense matrix kernel, and this kernel acts on the neighboring node latent state before aggregation. This is the graph neural operator part of the model and captures topology-aware local member interactions.

The `GlobalAttentionBlock` computes a scalar attention score for each node, normalizes the scores over each graph, forms a graph-level context vector by weighted summation of node embeddings, and broadcasts this context back to all nodes through a residual update. This adds nonlocal information exchange across the full structural graph, helping the model represent global load-transfer paths and boundary-condition effects.

In short, AGNO uses local graph neural operator updates to model member-level interactions and global attention updates to model graph-level structural coupling.

## Repository Layout

```text
AGNO/
|-- configs/                     # YAML experiment configurations
|-- data/                        # Included small benchmark datasets and MATLAB generators
|   |-- frame/                   # Crane-frame datasets
|   |-- truss/                   # 10-bar, bridge, and 4x4 grid-truss datasets
|   `-- MATLAB_code/             # MATLAB source for generating larger datasets
|-- docs/                        # Selected public figures and documentation assets
|-- lib/                         # Models, data loaders, training utilities
|-- res/
|   |-- plots/                   # Generated visualization outputs
|   `-- saved_models/            # Model checkpoints and cached test data
|-- graph_*_training.py          # GNN, AGNN, GNO, AGNO entry points
|-- GANO_*_training.py           # GANO baseline entry points
|-- requirements.txt             # Python dependencies
`-- LICENSE
```

## Models

The repository includes the models compared in the paper.

| Model | Description |
| --- | --- |
| `GNN` | Standard message-passing graph neural network baseline |
| `AGNN` | GNN with global attention |
| `GNO` | Graph neural operator with edge-conditioned kernel message passing |
| `AGNO` | Proposed local-global graph neural operator with global attention |
| `GANO` | Geometry-aware neural operator baseline |

The graph-based models are implemented in `lib/model_graph.py`. The GANO baselines for truss and frame systems are implemented in `lib/model_truss.py`, `lib/model_spacetruss.py`, `lib/model_frame.py`, and `lib/model_spaceframe.py`.

## Included Data

Small and medium benchmark datasets are included as MATLAB `.mat` files:

| Problem | Graph-model data | GANO data |
| --- | --- | --- |
| Crane plane frame | `data/frame/craneplaneGframe_9nodes_v2.mat`, `data/frame/craneplaneGframe_17nodes_v2.mat`, `data/frame/craneplaneGframe_33nodes_v2.mat` | `data/frame/craneplaneframe_load_9nodes_v2.mat`, `data/frame/craneplaneframe_load_17nodes_v2.mat`, `data/frame/craneplaneframe_load_33nodes_v2.mat` |
| 10-bar plane truss | `data/truss/10barplaneGtruss_load_v2.mat` | `data/truss/10barplanetruss_load_v2.mat` |
| Burro Creek bridge | `data/truss/69barplaneGtruss_load_v2.mat` | `data/truss/69barplanetruss_load_v2.mat` |
| Grid space truss, DS-4x4 | `data/truss/4x4Gtruss_load_v2.mat` | `data/truss/4x4spacetruss_load_v2.mat` |

Larger benchmark datasets are not stored directly in this repository. MATLAB source code for generating the larger problems is provided under `data/MATLAB_code/`.

Each training script reads the dataset path from the corresponding YAML file under `configs/`. Before running an experiment, check the `data.datapath` field and update it if needed:

```yaml
data:
  datapath: './data/truss/10barplaneGtruss_load_v2.mat'
```

## Pretrained Checkpoints

The best AGNO checkpoints for four representative examples are provided in `res/saved_models/`. These checkpoints correspond to the results reported in the study and were obtained from training runs of 1000 epochs.

| Checkpoint | Paper example / benchmark |
| --- | --- |
| `best_model_planeframe_AGNO.pkl` | Crane plane frame, MR-2 case in Example 6.1 |
| `best_model_planetruss_AGNO.pkl` | 10-bar plane truss |
| `best_model_spaceframe_AGNO.pkl` | Curved-roof space frame |
| `best_model_spacetruss_AGNO.pkl` | Grid space truss, DS-4x4 case |

These checkpoints are loaded automatically when running the corresponding AGNO script with `--phase test`, provided that the matching dataset path in the YAML config is available.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The requirements include PyTorch and PyTorch Geometric. If your CUDA version requires a specific PyTorch/PyG wheel, install the matching wheels first, then install the remaining packages from `requirements.txt`.

## Running Experiments

All entry-point scripts use the same basic arguments:

| Argument | Description |
| --- | --- |
| `--phase` | `train` for training, `test` for evaluation with a saved checkpoint |
| `--data` | Dataset family: `planetruss`, `spacetruss`, `planeframe`, or `spaceframe` |
| `--model` | Model name: `GNN`, `AGNN`, `GNO`, `AGNO`, or `GANO` |

Train AGNO on the 10-bar plane truss dataset:

```bash
python graph_planetruss_training.py --phase train --data planetruss --model AGNO
```

Test the trained AGNO checkpoint:

```bash
python graph_planetruss_training.py --phase test --data planetruss --model AGNO
```

Train AGNO on the DS-4x4 grid space truss dataset:

```bash
python graph_spacetruss_training.py --phase train --data spacetruss --model AGNO
```

Train AGNO on the crane plane-frame dataset:

```bash
python graph_planeframe_training.py --phase train --data planeframe --model AGNO
```

Train baseline graph models by changing `--model`:

```bash
python graph_planetruss_training.py --phase train --data planetruss --model GNO
python graph_planetruss_training.py --phase train --data planetruss --model GNN
python graph_planetruss_training.py --phase train --data planetruss --model AGNN
```

Train and test the GANO baseline:

```bash
python GANO_planetruss_training.py --phase train --data planetruss --model GANO
python GANO_planetruss_training.py --phase test --data planetruss --model GANO
```

The `test` phase expects the corresponding checkpoint to exist in `res/saved_models/`.

## Configuration

Each experiment is controlled by a YAML file named:

```text
configs/{MODEL}_{DATA}.yaml
```

Examples:

```text
configs/AGNO_planetruss.yaml
configs/GNO_spacetruss.yaml
configs/GANO_planeframe.yaml
```

Typical settings include:

| Section | Contents |
| --- | --- |
| `data` | Path to the `.mat` dataset |
| `model` | Input dimension, output dimension, hidden dimension, number of layers, edge feature dimension, dropout |
| `train` | Number of epochs, train/validation/test split sizes, batch size, learning rate, scheduler milestones, patience |

The graph-based AGNO/GNO/GNN/AGNN experiments use six layers and a latent dimension of 64 by default. The GANO baseline uses separate architecture settings following the baseline configuration used in the paper.

## Outputs

Training and evaluation outputs are written to:

```text
res/saved_models/
res/plots/
```

Common outputs include:

| Output | Description |
| --- | --- |
| `best_model_{data}_{model}.pkl` | Best checkpoint for graph-based models |
| `best_model_{geo_node}_{data}_{model}.pkl` | Best checkpoint for GANO |
| `evaluation_error_{data}_{model}.txt` | Evaluation errors |
| `epoch_####_last_attention_{data}_{model}.mat` | Saved attention weights for attention-based models |
| `test_cache_{data}_{model}.mat` | Cached test data and normalization metadata for post-processing |

## Reproducibility Notes

The entry-point scripts set a fixed random seed (`2025`) and enable deterministic PyTorch behavior where possible. Training results can still depend on GPU hardware, CUDA, PyTorch, and PyTorch Geometric versions.

The included `.mat` files were generated from linear elastic finite element simulations in MATLAB. For graph-based models, node features include normalized coordinates and applied loads, edge features include geometry-dependent member attributes, and outputs are nodal displacements or nodal displacement/rotation components depending on the structural type.

## Citation

If this code is useful for your work, please cite the associated paper:

```bibtex
@article{vo2026agno,
  title   = {Attention-based graph neural operators for learning parametric response mappings in discrete structures},
  author  = {Vo, Duy-Trung and Lee, Jaehong},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  doi     = {10.1016/j.engappai.2026.115232},
  url     = {https://doi.org/10.1016/j.engappai.2026.115232}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
