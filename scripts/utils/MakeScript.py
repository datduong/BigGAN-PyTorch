

# import os,sys,re,pickle
# import pandas as pd 
# import numpy as np 
# from datetime import datetime

# script = """#!/bin/bash

# # ! make dataset into hdf5 format
# cd /data/duongdb/BigGAN-PyTorch/
# rm -rf data_name*
# python make_hdf5.py --dataset data_name --batch_size 32 --data_root main_dir/
# cd main_dir/
# mkdir main_dir/data/
# mv ILSVRC128.hdf5 main_dir/data/

# # ! computer inception score
# source /data/$USER/conda/etc/profile.d/conda.sh
# conda activate py37
# cd /data/duongdb/BigGAN-PyTorch/
# python calculate_inception_moments.py --dataset data_hdf5 --data_root main_dir/data

# # ! make output folder
# mkdir main_dir/weights/
# mkdir main_dir/samples/
# mkdir main_dir/logs/
# mkdir main_dir/weights/

# # ! copy weights from pretrained model
# # scp /data/duongdb/BigGAN-PyTorch/100k/* main_dir/weights/

# """

# os.chdir('/data/duongdb/BigGAN-PyTorch/scripts')

# data_name = 'Isic19'
# data_hdf5 = data_name+'_hdf5'
# main_dir = '/data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/'

# # data_name = 'NF1Recrop'
# # data_hdf5 = data_name+'_hdf5'
# # main_dir = '/data/duongdb/SkinConditionImages11052020/Recrop/'

# # data_name = 'NF1Zoom'
# # data_hdf5 = data_name+'_hdf5'
# # main_dir = '/data/duongdb/SkinConditionImages11052020/ZoomCenter/'

# script = re.sub('data_name',data_name,script)
# script = re.sub('data_hdf5',data_hdf5,script)
# script = re.sub('main_dir',main_dir,script)

# now = datetime.now() # current date and time
# scriptname = 'script-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
# fout = open(scriptname,'w')
# fout.write(script)
# fout.close()



# ! script to run on gpu


import os,sys,re,pickle
import pandas as pd 
import numpy as np 
from datetime import datetime

scriptbase="""#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

# sinteractive --time=1:00:00 --gres=gpu:p100:4 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:4 --mem=48g --cpus-per-task=32 launch_BigGAN_bs512x4_pretrain_var5.sh

cd /data/duongdb/BigGAN-PyTorch/

# ! original lr --G_lr 1e-4 --D_lr 4e-4

python train.py \
--base_root rootname \
--data_root rootname \
--dataset dataset_name --parallel --shuffle --num_workers 16 --batch_size batchsize --load_in_mem \
--num_epochs 50 \
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
--G_ch arch_size --D_ch arch_size \
--ema --use_ema --ema_start 2000 \
--test_every 3 --save_every 3 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--z_var variance \
--pretrain /data/duongdb/BigGAN-PyTorch/100k \
--augment \
--experiment_name_suffix veryveryweakaug \
--Y_sample '4,4,5,5,6,6,7,7,9,9,9,10,10,10,13,13' \
--Y_pair '4,4,4,4,5,5,5,5,6,6,7,7,7,7,9,9\t10,10,13,13,6,6,7,7,7,7,9,9,10,10,10,10' 

# \
# --up_labels '4,5,6,7,9,10,13'

"""

# '4,4,4,4,5,5,5,5,6,6,7,7,7,7,9,9\t10,10,13,13,6,6,7,7,7,7,9,9,10,10,10,10'
# '4,4,5,5,6,6,7,7,9,9,10,10,13,13'

os.chdir('/data/duongdb/BigGAN-PyTorch/scripts')

batchsize = 128 # ! 152 batch is okay. larger size is recommended... but does it really matter when our data is smaller ? 
arch_size = 96 # default for img net 96, don't have a smaller pretrained weight # ! not same as G_attn
variance = 1
dataset_name = { 
                # 'NF1Recrop_hdf5':'/data/duongdb/SkinConditionImages11052020/Recrop/', # _hdf5
                # 'NF1Zoom':'/data/duongdb/SkinConditionImages11052020/ZoomCenter/', 
                'NF1ZoomIsic19':'/data/duongdb/SkinConditionImages11052020/ZoomCenter/',
                # 'Isic19':'/data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/'
                }

count=0
for dataname in dataset_name: 
  #
  count = count+1
  print (dataname)
  script = re.sub('arch_size',str(arch_size),scriptbase)
  script = re.sub('batchsize',str(batchsize),script)
  script = re.sub('variance',str(variance),script)
  script = re.sub('dataset_name',dataname,script)
  script = re.sub('rootname',dataset_name[dataname],script)
  if '_hdf5' not in dataname: 
    script = re.sub ( ' --load_in_mem', '', script ) ## ! read from data to do aug. so don't load into mem.
  else: 
    script = re.sub ( ' --augment', '', script ) ## ! dont aug with hdf5, we just load into mem
    script = re.sub ( ' --experiment_name_suffix veryweakaug', '', script ) ## ! dont aug with hdf5, we just load into mem
  #
  now = datetime.now() # current date and time
  scriptname = 'script'+str(count)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
  fout = open(scriptname,'w')
  fout.write(script)
  fout.close()
  os.system('sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:4 --mem=48g --cpus-per-task=32 ' + scriptname )



# # ! create pairs to view
# labels = {'EverythingElse': 4, 'MA': 7, 'BKL': 2, 'IP': 6, 'AK': 0, 'TSC': 13, 'SCC': 12, 'BCC': 1, 'DF': 3, 'VASC': 14, 'HMI': 5, 'NF1': 10, 'MEL': 8, 'ML': 9, 'NV': 11}

# our = 'EverythingElse HMI IP MA ML NF1 TSC'.split() 
# pout = []
# for p in our: 
#   pout.append( labels[p] )

# # 
# pout = pout + pout 
# pout.sort() 
# pout = ','.join(str(p) for p in pout)
# pout

# pairs = [ 'HMI IP', 'HMI MA', 'IP MA', 'MA ML', 'ML NF1', 'MA NF1', 'EverythingElse NF1', 'EverythingElse TSC']
# pairs = pairs + pairs
# pairs.sort() 

# p1 = []
# p2 = []
# for p in pairs: 
#   p = p.split()
#   p1.append (labels[p[0]]) 
#   p2.append (labels[p[1]])

# #
# pout = ','.join(str(i) for i in p1)+'\t'+','.join(str(i) for i in p2) 



