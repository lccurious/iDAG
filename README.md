# iDAG: Towards Invariant Causal Discovery for Domain Generalization

Official PyTorch implementation of iDAG

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for our study.

```
Python: 3.9.13
PyTorch: 1.10.0
Torchvision: 0.11.0
CUDA: 10.2
CUDNN: 7605
NumPy: 1.23.1
PIL: 9.2.0
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset PACS --data_dir /my/datasets/path
```

Experiment results are reported as a table. In the table, the row `SWAD` indicates out-of-domain accuracy from SWAD.
The row `SWAD (inD)` indicates in-domain validation accuracy.

Example results:
```
+------------+--------------+---------+---------+---------+---------+
| Selection  | art_painting | cartoon |  photo  |  sketch |   Avg.  |
+------------+--------------+---------+---------+---------+---------+
|   oracle   |   87.309%    | 85.181% | 96.781% | 83.174% | 88.112% |
|    iid     |   84.137%    | 78.891% | 96.108% | 77.449% | 84.146% |
|    last    |   82.428%    | 76.226% | 94.611% | 78.690% | 82.989% |
| last (inD) |   96.548%    | 95.338% | 95.523% | 95.715% | 95.781% |
| iid (inD)  |   97.301%    | 96.736% | 96.787% | 97.098% | 96.980% |
|    SWAD    |   90.811%    | 84.488% | 97.979% | 82.761% | 89.009% |
| SWAD (inD) |   98.054%    | 97.561% | 97.425% | 97.992% | 97.758% |
+------------+--------------+---------+---------+---------+---------+
```
In this example, the DG performance of SWAD for PACS dataset is 89.009%.

If you set `indomain_test` option to `True`, the validation set is splitted to validation and test sets,
and the `(inD)` keys become to indicate in-domain test accuracy.


### Reproduce the results of the paper

We provide the instructions to reproduce the main results of the paper, Table 1 and 2.
Note that the difference in a detailed environment or uncontrolled randomness may bring a little different result from the paper.

- PACS

```
python train_all.py PACS0 --dataset PACS --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py PACS1 --dataset PACS --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py PACS2 --dataset PACS --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir /my/datasets/path
```

- VLCS

```
python train_all.py VLCS0 --dataset VLCS --deterministic --trial_seed 0 --checkpoint_freq 50 --tolerance_ratio 0.2 --data_dir /my/datasets/path
python train_all.py VLCS1 --dataset VLCS --deterministic --trial_seed 1 --checkpoint_freq 50 --tolerance_ratio 0.2 --data_dir /my/datasets/path
python train_all.py VLCS2 --dataset VLCS --deterministic --trial_seed 2 --checkpoint_freq 50 --tolerance_ratio 0.2 --data_dir /my/datasets/path
```

- OfficeHome

```
python train_all.py OH0 --dataset OfficeHome --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py OH1 --dataset OfficeHome --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py OH2 --dataset OfficeHome --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir /my/datasets/path
```

- TerraIncognita

```
python train_all.py TR0 --dataset TerraIncognita --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py TR1 --dataset TerraIncognita --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py TR2 --dataset TerraIncognita --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir /my/datasets/path
```

- DomainNet

```
python train_all.py DN0 --dataset DomainNet --deterministic --trial_seed 0 --checkpoint_freq 500 --data_dir /my/datasets/path
python train_all.py DN1 --dataset DomainNet --deterministic --trial_seed 1 --checkpoint_freq 500 --data_dir /my/datasets/path
python train_all.py DN2 --dataset DomainNet --deterministic --trial_seed 2 --checkpoint_freq 500 --data_dir /my/datasets/path
```


## Main Results


## License

This source code is released under the MIT license, included [here](./LICENSE).

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.
