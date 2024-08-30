project_name="6g_23rd_August"
# nocompression-baseline
train_classifier_separetely_options=(True False)
train_snr_dbs=(0 random)

for TCS in ${train_classifier_separetely_options[@]}; do
    for train_snr in ${train_snr_dbs[@]}; do
        
        python ltrain/train_baselines.py compressor=nocompression train_classifier_separetely=$TCS train_snr_db=$train_snr logger.wandb_project=$project_name
           
    done  
done