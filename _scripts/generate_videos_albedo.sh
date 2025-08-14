#!/bin/bash
network_pkl=models/network-snapshot.pkl

lighting_pattern='albedo'
seeds=10-14 #ID

resolution=512
trunc=0.7
opacity_ones=False
grid=1x1
with_bg=False
outdir=out/${lighting_pattern}

CUDA_VISIBLE_DEVICES=0 python gen_videos_gsparams.py --outdir=${outdir}/ --trunc=${trunc} --seeds=${seeds} --grid=${grid} \
    --network=${network_pkl} --image_mode=image --g_type=G_ema --load_architecture=False \
    --nrr=${resolution} --opacity_ones=${opacity_ones} --lighting_pattern=${lighting_pattern}\
    --with_bg=${with_bg} --load_architecture=True \
