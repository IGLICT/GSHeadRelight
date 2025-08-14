# network_pkl=dummy
network_pkl=models/network-snapshot.pkl
dataset_path=/path/to/dataset

python calc_metrics.py --metrics=fid50k_full \
    --network=${network_pkl} --data=${dataset_path}


