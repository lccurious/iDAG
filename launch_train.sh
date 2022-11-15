# run HP search on OfficeHome @ RTX2080Ti
python -m domainbed.scripts.sweep launch \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep/officehome \
       --command_launcher multi_available_gpu \
       --algorithms DAGDG \
       --mem_usage 11GiB \
       --num_parallel 8 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3 \
       --hparams '{"resnet18": "True"}'

# delete incomplete OfficeHome @ RTX2080Ti
python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir=/home/hzn/datasets \
       --output_dir=train_output/sweep/officehome \
       --command_launcher multi_available_gpu \
       --algorithms DAGDG \
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
       --algorithms DAGDG \
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
       --output_dir train_output/sweep-v1/officehome \
       --command_launcher multi_available_gpu \
       --algorithms DAGDG \
       --mem_usage 10.5GiB \
       --num_parallel 8 \
       --launch_delay 5 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams 5 \
       --n_trials 3

# delete incomplete HP search on OfficeHome @ Zhejiang-2
python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir /data/huangzenan/SWAD/data \
       --output_dir train_output/sweep-v1/officehome \
       --command_launcher multi_available_gpu \
       --algorithms DAGDG \
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
       --output_dir train_output/sweep-v1/domainnet \
       --command_launcher multi_available_gpu \
       --algorithms DAGDG \
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
       --output_dir train_output/sweep-v1/domainnet \
       --command_launcher multi_available_gpu \
       --algorithms DAGDG \
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
       --algorithms DAGDG \
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
       --algorithms DAGDG \
       --mem_usage 11GiB \
       --num_parallel 8 \
       --launch_delay 3 \
       --datasets OfficeHome \
       --single_test_envs \
       --n_hparams_from 20 \
       --n_hparams 40 \
       --n_trials 3
