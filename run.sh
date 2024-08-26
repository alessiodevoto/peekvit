project_name="6g_23rd_August"

# AE-baseline
encoding_dims=(58 77 96 116 135 154 173 192)
train_classifier_separetely_options=(True False)
train_snr_dbs=(0 random)

for encoding_dim in ${encoding_dims[@]}; do
    for TCS in ${train_classifier_separetely_options[@]}; do
        for train_snr in ${train_snr_dbs[@]}; do
            python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=$encoding_dim train_classifier_separetely=$TCS train_snr_db=$train_snr logger.wandb_project=$project_name 
        done
    done
done

# tome-baseline
train_classifier_separetely_options=(True False)
train_snr_dbs=(0 random)
trace_losses=(False True)
r=( [2,2,2,2,2,2] [5,5,5,5,5,5] [10,10,10,10,10,10] [20,20,20,20,20,20] [30,30,30,30,30,30] [40,40,40,40,40,40]) 

for TCS in ${train_classifier_separetely_options[@]}; do
    for train_snr in ${train_snr_dbs[@]}; do
        for trace_loss in ${trace_losses[@]}; do
            for r_val in ${r[@]}; do
                python ltrain/train_baselines.py train_classifier_separetely=$TCS train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name
            done
        done
    done  
done


# pca-baseline
encoding_dims=(58 77 96 116 135 154 173 192)
train_classifier_separetely_options=(True False)
train_snr_dbs=(0 random)

for encoding_dim in ${encoding_dims[@]}; do
    for TCS in ${train_classifier_separetely_options[@]}; do
        for train_snr in ${train_snr_dbs[@]}; do
            python ltrain/train_baselines.py compressor=pca compressor.q=$encoding_dim train_classifier_separetely=$TCS train_snr_db=$train_snr logger.wandb_project=$project_name 
        done
    done
done