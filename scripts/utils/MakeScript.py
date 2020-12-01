import os,sys,re,pickle
import pandas as pd 
import numpy as np 
from datetime import datetime

script = """#!/bin/bash

# ! make dataset into hdf5 format
cd /data/duongdb/BigGAN-PyTorch/
rm -rf data_name*
python make_hdf5.py --dataset data_name --batch_size 32 --data_root main_dir/
cd main_dir/
mkdir main_dir/data/
mv ILSVRC128.hdf5 main_dir/data/

# ! computer inception score
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
cd /data/duongdb/BigGAN-PyTorch/
python calculate_inception_moments.py --dataset data_hdf5 --data_root main_dir/data

# ! make output folder
mkdir main_dir/weights/
mkdir main_dir/samples/
mkdir main_dir/logs/
mkdir main_dir/weights/

# ! copy weights from pretrained model
# scp /data/duongdb/BigGAN-PyTorch/100k/* main_dir/weights/

"""

os.chdir('/data/duongdb/BigGAN-PyTorch/scripts')

# data_name = 'Isic19'
# data_hdf5 = data_name+'_hdf5'
# main_dir = '/data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/'

data_name = 'NF1Recrop'
data_hdf5 = data_name+'_hdf5'
main_dir = '/data/duongdb/SkinConditionImages11052020/Recrop/'

# data_name = 'NF1Zoom'
# data_hdf5 = data_name+'_hdf5'
# main_dir = '/data/duongdb/SkinConditionImages11052020/ZoomCenter/'

script = re.sub('data_name',data_name,script)
script = re.sub('data_hdf5',data_hdf5,script)
script = re.sub('main_dir',main_dir,script)

now = datetime.now() # current date and time
scriptname = 'script-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
fout = open(scriptname,'w')
fout.write(script)
fout.close()



# ! script to run on gpu

script="""
#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

# sinteractive --time=1:00:00 --gres=gpu:p100:4 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2-12:00:00 --gres=gpu:p100:4 --mem=48g --cpus-per-task=32 launch_BigGAN_bs512x4_pretrain_var5.sh

cd /data/duongdb/BigGAN-PyTorch/

python train.py \
--base_root /data/duongdb/SkinConditionImages11052020/Recrop/ \
--data_root /data/duongdb/SkinConditionImages11052020/Recrop/ \
--dataset NF1Recrop_hdf5 --parallel --shuffle --num_workers 16 --batch_size 152 --load_in_mem  \
--num_G_accumulations 4 --num_D_accumulations 4 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 2000 \
--test_every 200 --save_every 200 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--resume \
--experiment_name_suffix Var10 \
--z_var 10

#! with 2019 skin cancer, use larger batch? 16 labels

# 725 484 8180 5113@128 4812@136 4675@140 4545@144 3896@168--fail 4306@152
# num_G_accumulations https://github.com/ajbrock/BigGAN-PyTorch/issues/70

"""