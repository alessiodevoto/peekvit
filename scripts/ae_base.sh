project_name="6g_December3"

# AE+gnn-classification
encoding_dims=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
train_snr_dbs=(random)

for encoding_dim in ${encoding_dims[@]}; do
        for train_snr in ${train_snr_dbs[@]}; do
            python ltrain/train_baselines.py\
            compressor=ae\
            compressor.encoding_dim=$encoding_dim\
            train_snr_db=$train_snr\
            logger.wandb_project=$project_name\
            classifier_num_blocks=1
    done
done


# python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=1.0 train_snr_db=random logger.wandb_project=test_6g classifier_num_blocks=1
# python ltrain/train_baselines.py compressor=ae_gnn compressor.encoding_dim=1.0 train_snr_db=random logger.wandb_project=test_6g classifier_num_blocks=1


# python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=1.0 train_snr_db=random logger.wandb_project=test_6g classifier_num_blocks=6
# python ltrain/train_baselines.py compressor=ae_gnn compressor.encoding_dim=1.0 train_snr_db=random logger.wandb_project=test_6g classifier_num_blocks=6


# python ltrain/train_baselines.py compressor=tome_baseline compressor.encoding_dim=1.0 train_snr_db=random logger.wandb_project=test_6g classifier_num_blocks=1
# python ltrain/train_baselines.py compressor=tome_gnn compressor.encoding_dim=1.0 train_snr_db=random logger.wandb_project=test_6g classifier_num_blocks=1

