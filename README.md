# AR-GCN

This is the official repository for **Revisiting Autoregressive GCNs for Vehicle Routing Problems** ([NeurIPS 2026 Poster](https://neurips.cc/virtual/2026/loc/sydney/poster/152174)).

## Installation

Use Linux, Python 3.10 or later, and PyTorch with NVIDIA CUDA support. From the repository root:

```bash
pip install -r requirements.txt
```

## Data Generation

`dataset/` already contains the released synthetic test sets, the TSPLIB/CVRPLIB sets and their labels, so evaluation runs out of the box. The scripts below regenerate that data, or produce training sets in the same format.

Synthetic test sets, written under `$DATASET_ROOT` (default `./dataset_generated`):

```bash
bash scripts/generator_SVRP_test.sh   # uniform / clustered / explosion / implosion, n = 100 ... 10000
bash scripts/generator_AVRP_test.sh   # TMAT distance matrices, n = 50 ... 1000
```

Every generator call takes the same `--distribution` flag: `uniform`, `clustered`, `explosion` and `implosion` write node coordinates as `nodes.npy`, while `tmat` writes a distance matrix as `edges.npy`.

Training sets, written to `dataset/SVRP_train/` and `dataset/AVRP_train/` (200k instances of 100 nodes):

```bash
bash scripts/generator_SVRP_train.sh
bash scripts/generator_AVRP_train.sh
```

LKH labels for training (`--baseline lkh` reads `tours_<PROBLEM>_lkh.npy`):

```bash
bash scripts/lkh_solver_train.sh                           # AVRP training set: ATSP and ACVRP
SYMMETRIC=1 PROBLEM=TSP  bash scripts/lkh_solver_train.sh  # SVRP training set: STSP
SYMMETRIC=1 PROBLEM=CVRP bash scripts/lkh_solver_train.sh  # SVRP training set: SCVRP
```

Labels for the test suites:

```bash
DATASET_ROOT=dataset bash scripts/lkh_solver_test.sh svrp  # STSP labels, tours_TSP_lkh.npy
DATASET_ROOT=dataset bash scripts/lkh_solver_test.sh avrp  # ATSP labels, tours_TSP_lkh.npy
DATASET_ROOT=dataset bash scripts/hgs_solver_test.sh all   # SCVRP labels, tours_CVRP_hgs.npy and ACVRP labels, tours_CVRP.npy
```

Every script reads the same kind of overrides: `DATASET_ROOT` wherever data is read or written, `PYTHON` to pick the interpreter, and the per-script knobs documented at the top of each file (`VRP_SIZE`, `DATA_SIZE`, `SCALE`, `MAX_THREADS`, `CKPT_ROOT`, `OUTPUT_DIR`, ...). The HGS solver calls `hygese`, which aborts when the scaled distances exceed 1e5, hence `SCALE=10000` in the scripts and in the solver default.

`lkh_solver*.sh` run the prebuilt Linux binary `data_factory/LKH-3.0.13/LKH`, so they need Linux. If that binary does not run on your machine, first check that it is executable (`chmod +x data_factory/LKH-3.0.13/LKH`); if it still fails, rebuild it from source following `data_factory/LKH-3.0.13/README.txt`:

```bash
cd data_factory/LKH-3.0.13
make                        # produces ./LKH
./LKH pr2392.par            # sanity check, solves a 2392-node instance and writes ./route
rm -f route
```

`make` needs a C compiler such as `gcc`; run `make clean` before rebuilding after you edit `SRC/Makefile` (for example to switch to the three-level tour tree). The rebuilt `LKH` replaces the shipped binary in place, and `lkh_solver.py` picks it up through `--lkh_file`, which defaults to that path.

## Evaluation

```bash
bash scripts/test.sh 0 small STSP-128-6-50-3-mean 16  # Try 16 Uniform-100 instances
bash scripts/test.sh 0 synthetic                     # All synthetic benchmarks
bash scripts/test.sh 0 lib                           # STSPLIB, ATSPLIB, and CVRPLIB X/XL
```

`0` is the GPU index. For synthetic tests, the optional fourth argument limits the number of instances. Checkpoints are read from `output/train/` (override with `CKPT_ROOT=...`), logs are written to `output/test/` (`OUTPUT_DIR=...`).

## Training

The presets use Uniform-100 or TMAT-100 instances with LKH labels; see [Data Generation](#data-generation) for producing them. Then choose one command:

```bash
bash scripts/train.sh 0 STSP-128-6-50-3-mean dataset output/retrain  # One preset
bash scripts/train.sh 0 all dataset output/retrain                   # All six presets
```

Preset names follow `problem-hidden_dim-layers-k-epochs-aggregation`. To select another preset, use its directory name under `output/train/`. New checkpoints go to `output/retrain/`.

## Custom Training

For example, train a symmetric TSP model with hidden dimension 256, six encoder layers, KNN with `k=50`, mean aggregation, and five epochs:

```bash
python -u AR_GCN/train/main.py \
  --problem TSP --symmetric \
  --d_model 256 --n_encoders 6 --max_k 50 --aggregation mean \
  --n_epoch 5 --batch_size 100 --learning_rate 5e-4 \
  --baseline lkh --data_dir dataset/SVRP_train/uniform/100 \
  --data_size 200000 --cuda_id 0 \
  --output_dir output/retrain --save_name STSP-256-6-50-5-mean
```

Change the flags to set the model and training parameters. `--data_size` sets the number of labeled instances, and `--save_name` sets the run directory name.

## Citation

We hope this repository is helpful for your work. We would be delighted if you give this repository a star and cite our paper:

```bibtex
@inproceedings{AR-GCN,
    title = {Revisiting Autoregressive {GCN}s for Vehicle Routing Problems},
    author = {Zhipeng Zhong and Junquan Huang and Yu Huang and Boyuan Zheng and Jungang Li and Zihao Dongfang and Song Dai and Xuming Hu and Zong-Gan Chen},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2026},
    url = {https://neurips.cc/virtual/2026/loc/sydney/poster/152174}
}
```
