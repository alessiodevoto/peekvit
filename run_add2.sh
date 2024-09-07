
# project_name="6g_4rd_September"
# # nocompression-baseline
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# #r=([5,5,5,5,5,5] [10, 10, 10, 10, 10, 10] [30,30,30,30,30,30] [40,40,40,40,40,40])
# r=([0,0,0,0,0,0])
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
# trace_losses=(False True)
# r=([0,0,0,0,0,0] [5,5,5,5,5,5] [10,10,10,10,10,10] [30,30,30,30,30,30] [40,40,40,40,40,40]) 

# for train_snr in ${train_snr_dbs[@]}; do
#     for trace_loss in ${trace_losses[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name plot_groups=True model_type=paralel_models
#         done
#     done
# done  


# encoding_dims=(58 77 96 116 135 154 173 192)
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)

# for encoding_dim in ${encoding_dims[@]}; do
#         for train_snr in ${train_snr_dbs[@]}; do
#             python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=$encoding_dim train_snr_db=$train_snr logger.wandb_project=$project_name model_type=paralel_models plot_groups=False
#     done
# done


project_name="6g_4rd_September"
encoding_dims=(58 135)
train_classifier_separetely_options=(False)
train_snr_dbs=(0 random)

for encoding_dim in ${encoding_dims[@]}; do
        for train_snr in ${train_snr_dbs[@]}; do
            python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=$encoding_dim train_snr_db=$train_snr logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=False
    done
done


# tome-ae-baseline
train_classifier_separetely_options=(False)
train_snr_dbs=(0 random)
rocinstruct_images_a=(True False)
trace_losses=(True)
encoding_dims=(135 173)
r=([5,5,5,5,5,5] [30,30,30,30,30,30]) 

for rec_img in ${rocinstruct_images_a[@]}; do
    for train_snr in ${train_snr_dbs[@]}; do
        for trace_loss in ${trace_losses[@]}; do
            for encoding_dim in ${encoding_dims[@]}; do
                for r_val in ${r[@]}; do
                    python ltrain/train_baselines.py compressor=tome_ae compressor.encoding_dim=$encoding_dim train_classifier_separetely=False train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=$rec_img
                done
            done
        done
    done  
done


# # tome-ae-baseline TAC 
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# rocinstruct_images_a=(False)
# trace_losses=(False)
# encoding_dims=(135 173)
# r=([10,10,10,10,10,10] [30,30,30,30,30,30]) 

# for rec_img in ${rocinstruct_images_a[@]}; do
#     for train_snr in ${train_snr_dbs[@]}; do
#         for trace_loss in ${trace_losses[@]}; do
#             for encoding_dim in ${encoding_dims[@]}; do
#                 for r_val in ${r[@]}; do
#                     python ltrain/train_baselines.py compressor=tome_ae compressor.encoding_dim=$encoding_dim train_classifier_separetely=False train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=$rec_img
#                 done
#             done
#         done
#     done  
# done



# # TAC
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# rocinstruct_images_a=(False)
# trace_losses=(False)
# encoding_dims=(173)
# r=([5,5,5,5,5,5]) 

# for rec_img in ${rocinstruct_images_a[@]}; do
#     for train_snr in ${train_snr_dbs[@]}; do
#         for trace_loss in ${trace_losses[@]}; do
#             for encoding_dim in ${encoding_dims[@]}; do
#                 for r_val in ${r[@]}; do
#                     python ltrain/train_baselines.py compressor=tome_ae compressor.encoding_dim=$encoding_dim train_classifier_separetely=False train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=$rec_img
#                 done
#             done
#         done
#     done  
# done