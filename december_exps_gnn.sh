project_name="6g_December2"

# AE+gnn-classification
encoding_dims=(86  96 105 115 124 134 144 153 163 172 182 192)
train_snr_dbs=(random 0)

for encoding_dim in ${encoding_dims[@]}; do
        for train_snr in ${train_snr_dbs[@]}; do
            python ltrain/train_baselines.py compressor=ae_gnn compressor.encoding_dim=$encoding_dim train_snr_db=$train_snr logger.wandb_project=$project_name model_type=gnn_classifier plot_groups=False reconstruct_images=False train_classifier_separetely=False
    done
done


