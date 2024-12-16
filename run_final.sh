# # TCR
# project_name="6g_4rd_September"
# # nocompression-baseline
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# r=([5,5,5,5,5,5] [30,30,30,30,30,30] [40,40,40,40,40,40])

# for TCS in ${train_classifier_separetely_options[@]}; do
#     for train_snr in ${train_snr_dbs[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome train_classifier_separetely=$TCS train_snr_db=$train_snr logger.wandb_project=$project_name compressor.r=$r_val plot_groups=True model_type=paralel_models reconstruct_images=True compressor.use_trace_loss=False
#         done
#     done  
# done


# # tome-ae-baseline TACR
# project_name="6g_4rd_September"
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# encoding_dims=(173)
# r=([10,10,10,10,10,10]) 
# for train_snr in ${train_snr_dbs[@]}; do
#     for encoding_dim in ${encoding_dims[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome_ae compressor.encoding_dim=$encoding_dim train_classifier_separetely=False train_snr_db=$train_snr compressor.use_trace_loss=False compressor.r=$r_val logger.wandb_project=$project_name model_type=paralel_models plot_groups=True reconstruct_images=True
#         done
#     done
# done  

# # TACR
# project_name="6g_4rd_September"
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# encoding_dims=(173)
# r=([10,10,10,10,10,10]) 
# for train_snr in ${train_snr_dbs[@]}; do
#     for encoding_dim in ${encoding_dims[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome_ae compressor.encoding_dim=$encoding_dim train_classifier_separetely=False train_snr_db=$train_snr compressor.use_trace_loss=False compressor.r=$r_val logger.wandb_project=$project_name model_type=paralel_models plot_groups=True reconstruct_images=False
#         done
#     done
# done  


# # ACR
# project_name="6g_7th_September"
# encoding_dims=(58 135 163)
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# for encoding_dim in ${encoding_dims[@]}; do
#         for train_snr in ${train_snr_dbs[@]}; do
#             python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=$encoding_dim train_snr_db=$train_snr logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=True
#     done
# done


# # AC
# project_name="6g_7th_September"
# encoding_dims=(23 18)
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)

# for encoding_dim in ${encoding_dims[@]}; do
#         for train_snr in ${train_snr_dbs[@]}; do
#             python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=$encoding_dim train_snr_db=$train_snr logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=False
#     done
# done



# # ACR
# project_name="6g_7th_September"
# encoding_dims=(23 18)
# train_classifier_separetely_options=(False)
# train_snr_dbs=(0 random)
# for encoding_dim in ${encoding_dims[@]}; do
#         for train_snr in ${train_snr_dbs[@]}; do
#             python ltrain/train_baselines.py compressor=ae compressor.encoding_dim=$encoding_dim train_snr_db=$train_snr logger.wandb_project=$project_name model_type=paralel_models plot_groups=False reconstruct_images=True
#     done
# done


project_name="6g_plots_September"
# tome-baseline TC
train_classifier_separetely_options=(False)
train_snr_dbs=(random)
trace_losses=(False)
r=([5,5,5,5,5,5] [10,10,10,10,10,10] [30,30,30,30,30,30]) 

for train_snr in ${train_snr_dbs[@]}; do
    for trace_loss in ${trace_losses[@]}; do
        for r_val in ${r[@]}; do
            python ltrain/train_baselines.py compressor=tome train_snr_db=$train_snr compressor.use_trace_loss=$trace_loss compressor.r=$r_val logger.wandb_project=$project_name plot_groups=True model_type=paralel_models reconstruct_images=False
        done
    done
done

# # TCR
# project_name="6g_plots_September"
# train_classifier_separetely_options=(False)
# train_snr_dbs=(random)
# r=([5,5,5,5,5,5] [10,10,10,10,10,10] [30,30,30,30,30,30])

# for TCS in ${train_classifier_separetely_options[@]}; do
#     for train_snr in ${train_snr_dbs[@]}; do
#         for r_val in ${r[@]}; do
#             python ltrain/train_baselines.py compressor=tome train_classifier_separetely=$TCS train_snr_db=$train_snr logger.wandb_project=$project_name compressor.r=$r_val plot_groups=True model_type=paralel_models reconstruct_images=True compressor.use_trace_loss=False
#         done
#     done  
# done

