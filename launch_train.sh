# run HP search on OfficeHome @ RTX2080Ti
python -m domainbed.scripts.sweep launch \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep-v2/officehome \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 5GiB \
       --num_parallel 8 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 20 \
       --n_trials 3 \
       --hparams '{"resnet18": "True", "hidden_size": 512, "out_dim": 512, "checkpoint_freq": 300, "steps": 3000}'

# run HP search on OfficeHome @ RTX2080Ti with accumulation gradients
python -m domainbed.scripts.sweep launch \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep-v2/officehome \
       --command_launcher multi_available_gpu \
       --algorithms iDAGamp \
       --mem_usage 10GiB \
       --num_parallel 8 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 20 \
       --n_trials 3 \
       --steps 6000 \
       --checkpoint_freq 600 \
       --hparams '{"hidden_size": 512, "out_dim": 512, "batch_size":16}'

# run HP search on VLCS @ RTX2080Ti with accumulation gradients
python -m domainbed.scripts.sweep launch \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep-v2/VLCS \
       --command_launcher multi_available_gpu \
       --algorithms iDAGamp \
       --mem_usage 10GiB \
       --num_parallel 8 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 20 \
       --n_trials 3 \
       --steps 6000 \
       --checkpoint_freq 600 \
       --hparams '{"hidden_size": 512, "out_dim": 512, "batch_size":16}'

# run HP search on OfficeHome @ RTX2080Ti
python -m domainbed.scripts.sweep launch \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep-v2/officehome \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 10GiB \
       --num_parallel 8 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 20 \
       --n_trials 3 \
       --hparams '{"hidden_size": 512, "out_dim": 512, "checkpoint_freq": 300, "steps": 3000}'

# run HP search on OfficeHome @ RTX2080Ti
python -m domainbed.scripts.sweep launch \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep-v2/pacs \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 5GiB \
       --num_parallel 8 \
       --datasets PACS \
       --single_test_envs \
       --n_hparams 20 \
       --n_trials 3 \
       --hparams '{"hidden_size": 512, "out_dim": 512}'

# delete incomplete OfficeHome @ RTX2080Ti
python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep-v2/officehome \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 11GiB \
       --num_parallel 8 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3 \
       --hparams '{"resnet18": "True"}'

# run HP search on PACS @ RTX2080Ti
CUDA_VISIBLE_DEVICES=0,1 python -m domainbed.scripts.sweep launch \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep-v1/pacs \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 4GiB \
       --num_parallel 4 \
       --datasets PACS \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3 \
       --hparams '{"resnet18": "True"}'

# run HP search on OfficeHome @ Zhejiang-2
export CUDA_VISIBLE_DEVICES=0,2,3
python -m domainbed.scripts.sweep launch \
       --data_dir /data/huangzenan/SWAD/data \
       --output_dir train_output/sweep-v2/officehome \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 11GiB \
       --num_parallel 8 \
       --launch_delay 5 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3

# delete incomplete HP search on OfficeHome @ Zhejiang-2
python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir /data/huangzenan/SWAD/data \
       --output_dir train_output/sweep-v2/officehome \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 11GiB \
       --num_parallel 8 \
       --launch_delay 5 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3

# run HP search on DomainNet @ Zhejiang-2
python -m domainbed.scripts.sweep launch \
       --data_dir /data/huangzenan/SWAD/data \
       --output_dir train_output/sweep-v2/domainnet \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 17GiB \
       --num_parallel 8 \
       --launch_delay 3 \
       --datasets DomainNet \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3

# delete incomplete HP search on DomainNet @ Zhejiang-2
python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir /data/huangzenan/SWAD/data \
       --output_dir train_output/sweep-v2/domainnet \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 17GiB \
       --num_parallel 8 \
       --launch_delay 3 \
       --datasets DomainNet \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3

# run HP search on DomainNet @ Zhejiang-5
python -m domainbed.scripts.sweep launch \
       --data_dir /data/hzn/TransferLearning/SWAD/data \
       --output_dir train_output/sweep-v1/domainnet \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 17GiB \
       --num_parallel 8 \
       --launch_delay 3 \
       --datasets DomainNet \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3

# run HP search on OfficeHome @ Zhejiang-5
python -m domainbed.scripts.sweep launch \
       --data_dir /data/hzn/TransferLearning/SWAD/data \
       --output_dir train_output/sweep-v1/officehome \
       --command_launcher multi_available_gpu \
       --algorithms iDAG \
       --mem_usage 11GiB \
       --num_parallel 8 \
       --launch_delay 3 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams_from 20 \
       --n_hparams 40 \
       --n_trials 3 \
       --hparams '{"checkpoint_freq": 300, "steps": 3000}'
