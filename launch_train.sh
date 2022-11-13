python -m domainbed.scripts.sweep $1 \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep/officehome \
       --command_launcher multi_available_gpu \
       --algorithms DAGDG \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 1 \
       --hparams '{"resnet18": "True"}'
