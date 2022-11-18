# run HP search on OfficeHome @ RTX2080Ti
python -m domainbed.scripts.sweep launch \
       --data_dir=/my/datasets/path \
       --output_dir=train_output/sweep-v3/officehome \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 12GiB \
       --num_parallel 8 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 20 \
       --n_trials 3 \
       --hparams '{"hidden_size": 512, "out_dim": 512, "checkpoint_freq": 300, "steps": 3000}'

# Single run
python train_all.py OH0 \
  --algorithm iDAG \
  --data_dir /my/datasets/path \
  --dataset OfficeHome \
  --holdout_fraction 0.2 \
  --output_dir train_output/officehome \
  --seed 0 \
  --test_envs 2 \
  --trial_seed 0 \
  --hparams {"hidden_size": 512, "out_dim": 512, "checkpoint_freq": 300, "steps": 3000}

python train_all.py CMNIST0 \
  --algorithm iDAGCMNIST \
  --data_dir /my/datasets/path \
  --dataset ColoredMNIST \
  --holdout_fraction 0.2 \
  --output_dir train_output/cmnist \
  --seed 0 \
  --test_envs 0 \
  --trial_seed 0
