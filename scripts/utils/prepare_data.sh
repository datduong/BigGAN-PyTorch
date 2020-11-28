#!/bin/bash
python make_hdf5.py --dataset I128 --batch_size 256 --data_root data
python calculate_inception_moments.py --dataset I128_hdf5 --data_root data


# ! my own stuffs. 
cd /data/duongdb/BigGAN-PyTorch/
rm -rf NF1_i*
python make_hdf5.py --dataset NF1 --batch_size 64 --data_root /data/duongdb/SkinConditionImages11052020/Recrop/
cd /data/duongdb/SkinConditionImages11052020/Recrop/
mv ILSVRC128.hdf5 /data/duongdb/SkinConditionImages11052020/Recrop/data/

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
cd /data/duongdb/BigGAN-PyTorch/
python calculate_inception_moments.py --dataset NF1_hdf5 --data_root /data/duongdb/SkinConditionImages11052020/Recrop/data


