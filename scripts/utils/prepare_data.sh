#!/bin/bash
python make_hdf5.py --dataset I128 --batch_size 256 --data_root data
python calculate_inception_moments.py --dataset I128_hdf5 --data_root data


# ! my own stuffs. 
cd /data/duongdb/BigGAN-PyTorch/
rm -rf NF1Recrop*
python make_hdf5.py --dataset NF1Recrop --batch_size 64 --data_root /data/duongdb/SkinConditionImages11052020/Recrop/
cd /data/duongdb/SkinConditionImages11052020/Recrop/
mv ILSVRC128.hdf5 /data/duongdb/SkinConditionImages11052020/Recrop/data/

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
cd /data/duongdb/BigGAN-PyTorch/
python calculate_inception_moments.py --dataset NF1Recrop_hdf5 --data_root /data/duongdb/SkinConditionImages11052020/Recrop/data

mkdir /data/duongdb/SkinConditionImages11052020/Recrop/weights/Var10
scp /data/duongdb/BigGAN-PyTorch/100k/* /data/duongdb/SkinConditionImages11052020/Recrop/weights/Var10





# ! NF1 zoom+center, no isic19
cd /data/duongdb/BigGAN-PyTorch/
rm -rf NF1Zoom*
python make_hdf5.py --dataset NF1Zoom --batch_size 64 --data_root /data/duongdb/SkinConditionImages11052020/ZoomCenter/
cd /data/duongdb/SkinConditionImages11052020/ZoomCenter/
mkdir /data/duongdb/SkinConditionImages11052020/ZoomCenter/data/
mv ILSVRC128.hdf5 /data/duongdb/SkinConditionImages11052020/ZoomCenter/data/

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
cd /data/duongdb/BigGAN-PyTorch/
python calculate_inception_moments.py --dataset NF1Zoom_hdf5 --data_root /data/duongdb/SkinConditionImages11052020/ZoomCenter/data

mkdir /data/duongdb/SkinConditionImages11052020/ZoomCenter/weights/Var10
scp /data/duongdb/BigGAN-PyTorch/100k/* /data/duongdb/SkinConditionImages11052020/ZoomCenter/weights/Var10




# ! just only isic19
cd /data/duongdb/BigGAN-PyTorch/
rm -rf Isic19*
python make_hdf5.py --dataset Isic19 --batch_size 64 --data_root /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/
cd /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/
mkdir /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/data/
mv ILSVRC128.hdf5 /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/data/

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
cd /data/duongdb/BigGAN-PyTorch/
python calculate_inception_moments.py --dataset Isic19_hdf5 --data_root /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/data

mkdir /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/weights/
mkdir /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/samples/
mkdir /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/logs/

mkdir /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/weights/Var10
scp /data/duongdb/BigGAN-PyTorch/100k/* /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/weights/Var10


# ! combine isic2019 with zoom+center images. 
ourdata=/data/duongdb/SkinConditionImages11052020/ZoomCenter/
# scp -r $ourdata/OneLabelOneFolder/* $ourdata/OneLabelOneFolderWithIsic19
# scp -r /data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte/jpeg-isic2019-512x512/OneLabelOneFolder/* $ourdata/OneLabelOneFolderWithIsic19

cd /data/duongdb/BigGAN-PyTorch/
rm -rf NF1ZoomIsic19*
python make_hdf5.py --dataset NF1ZoomIsic19 --batch_size 64 --data_root $ourdata > NF1ZoomIsic19_makedata.log

cd $ourdata
mkdir $ourdata/data/
mv ILSVRC128.hdf5 $ourdata/data/

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
cd /data/duongdb/BigGAN-PyTorch/
python calculate_inception_moments.py --dataset NF1ZoomIsic19_hdf5 --data_root $ourdata/data/

mkdir $ourdata/weights/
mkdir $ourdata/samples/
mkdir $ourdata/logs/

