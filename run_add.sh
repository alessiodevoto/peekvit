# project_name="6g_4rd_September"
# # nocompression-baseline
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# #r=([5,5,5,5,5,5] [10, 10, 10, 10, 10, 10] [30,30,30,30,30,30] [40,40,40,40,40,40])
# r=([10,10,10,10,10,10])
# for TCS in ${train_classifier_separetely_options[@]}; do
#     for train_snr in ${train_snr_dbs[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome_onlyclf train_classifier_separetely=$TCS train_snr_db=$train_snr logger.wandb_project=$project_name compressor.r=$r_val reconstruct_images=False plot_groups=False
#         done
#     done  
# done

# project_name="6g_4rd_September"
# # tome-baseline
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# trace_losses=(False)
# r=([0,0,0,0,0,0]) 

# for train_snr in ${train_snr_dbs[@]}; do
#     for trace_loss in ${trace_losses[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name plot_groups=True model_type=paralel_models
#         done
#     done
# done  


# Finish the failed runs

project_name="6g_4rd_September"
# tome-baseline TC
train_classifier_separetely_options=(False)
train_snr_dbs=(0 random)
trace_losses=(False)
r=([0,0,0,0,0,0]) 

for train_snr in ${train_snr_dbs[@]}; do
    for trace_loss in ${trace_losses[@]}; do
        for r_val in ${r[@]}; do
            python ltrain/train_baselines.py compressor=tome train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name plot_groups=False model_type=paralel_models reconstruct_images=False
        done
    done
done  

project_name="6g_4rd_September"
# tome-baseline TCR
train_classifier_separetely_options=(False)
train_snr_dbs=(0 random)
trace_losses=(False)
r=([0,0,0,0,0,0]) 

for train_snr in ${train_snr_dbs[@]}; do
    for trace_loss in ${trace_losses[@]}; do
        for r_val in ${r[@]}; do
            python ltrain/train_baselines.py compressor=tome train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name plot_groups=False model_type=paralel_models reconstruct_images=True
        done
    done
done  


# project_name="6g_4rd_September"
# # tome-baseline
# train_classifier_separetely_options=(False)
# train_snr_dbs=(random)
# trace_losses=(False True)
# r=([5,5,5,5,5,5] [10,10,10,10,10,10] [30,30,30,30,30,30] [40,40,40,40,40,40]) 

# for train_snr in ${train_snr_dbs[@]}; do
#     for trace_loss in ${trace_losses[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name plot_groups=True model_type=paralel_models
#         done
#     done
# done  

# project_name="6g_4rd_September"
# # tome-baseline
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# trace_losses=(False)
# r=([0,0,0,0,0,0]) 

# for train_snr in ${train_snr_dbs[@]}; do
#     for trace_loss in ${trace_losses[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name plot_groups=False model_type=paralel_models
#         done
#     done
# done 
