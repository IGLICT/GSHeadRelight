export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0

# hyper-parameters
kimg=12000
gen_pose_cond=True
use_pe=True
metric=fid2k_full # fid2k-full for training
blur_fade_kimg=200
gamma=1
gpc_reg_prob=0.5
center_dists=1.0
prob_uniform=0.5 # only use for FFHQ
res_end=256
num_pts=256
nrr=512

expname=train
dataset=FFHQ512
dataset_path=/path/to/dataset
outdir=logs/${dataset}/${expname}
sh_dir=asset/example_light
ngpus=4
batchsize=32
batch_gpu=8
snap=20
rgb_sh=False
with_bg=False

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --outdir=${outdir} --sh_dir=${sh_dir} --cfg=ffhq --data=${dataset_path} \
  --gpus=${ngpus} --batch=${batchsize} --batch-gpu=${batch_gpu} --gamma=${gamma} --gen_pose_cond=${gen_pose_cond} --neural_rendering_resolution_initial=${nrr} --metrics=${metric} --blur_fade_kimg=${blur_fade_kimg} \
  --kimg=${kimg} --gaussian_num_pts=${num_pts} --start_pe=${use_pe} --gpc_reg_prob=${gpc_reg_prob} \
  --center_dists=${center_dists} --prob_uniform=${prob_uniform} --res_end=${res_end} --snap=${snap} --rgb_sh=${rgb_sh} --with_bg=${with_bg} \