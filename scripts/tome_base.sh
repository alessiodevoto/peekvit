project_name="6g_December3"


train_snr_dbs=(random)
#rs=([5,5,5,5,5,5] [10,10,10,10,10,10] [30,30,30,30,30,30] [40,40,40,40,40,40])
rs=([10,10,10,10,10,10])

for r in ${rs[@]}; do
        for train_snr in ${train_snr_dbs[@]}; do
            python ltrain/train_baselines.py\
            compressor=tome_baseline\
            compressor.encoding_dim=1.0\
            compressor.r=$r\
            train_snr_db=$train_snr\
            logger.wandb_project=$project_name\
            classifier_num_blocks=1
    done
done



#python ltrain/train_baselines.py compressor=tome_baseline compressor.encoding_dim=1.0 train_snr_db=random logger.wandb_project=test_6g classifier_num_blocks=1
#python ltrain/train_baselines.py compressor=tome_gnn compressor.encoding_dim=1.0 train_snr_db=random logger.wandb_project=test_6g classifier_num_blocks=1

