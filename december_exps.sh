project_name="6g_December"

# AE-classification
encoding_dims=(86  96 105 115 124 134 144 153 163 172 182 192)
train_snr_dbs=(random)

for encoding_dim in ${encoding_dims[@]}; do
        for train_snr in ${train_snr_dbs[@]}; do
            python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=$encoding_dim train_snr_db=$train_snr logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=False
    done
done

# AE-classification (No noise during training)
encoding_dims=(9  19  28  38  48  57  67  76  86  96 105 115 124 134 144 153 163 172 182 192)
train_snr_dbs=(0)

for encoding_dim in ${encoding_dims[@]}; do
        for train_snr in ${train_snr_dbs[@]}; do
            python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=$encoding_dim train_snr_db=$train_snr logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=False
    done
done
