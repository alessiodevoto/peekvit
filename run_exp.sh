# reconstruct_images=True
# transmit_cls_token=False
# trace_loss=(False True)
# validate_snr=(-10 -5 0 5 10)
# train_snr_dbs=("random" 0)
# classfy_images=(True False)
# r=( [0,0,0,0,0,0] [5,5,5,5,5,5] [10,10,10,10,10,10] [20,20,20,20,20,20] [30,30,30,30,30,30] [40,40,40,40,40,40] [100,100,100,100,100,100])
# model_type=('Encoder_Decoder_two_models_sequential') 

# for trace in ${trace_loss[@]}; do
#     for r_val in ${r[@]}; do
#         for model in ${model_type[@]}; do
#             for snr in ${validate_snr[@]}; do
#                 for train_snr in ${train_snr_dbs[@]}; do
#                     for classfy_images in ${classfy_images[@]}; do
#                         python ltrain/train.py decoder.use_trace_loss=$trace encoder.r=$r_val model_type=$model logger.wandb_project=6g_12th_July reconstruct_images=$reconstruct_images validate_snr_db=$snr transmit_cls_token=$transmit_cls_token train_snr_db=$train_snr classfy_images=$classfy_images
#                     done
#                 done
#             done
#         done
#     done
# done


reconstruct_images=True
transmit_cls_token=False
trace_loss=(False True)
validate_snr=(-10 -5 0 5 10)
train_snr_dbs=(0 random)
classfy_images=(True False)
r=( [2,2,2,2,2,2] [5,5,5,5,5,5] [10,10,10,10,10,10] [20,20,20,20,20,20] [30,30,30,30,30,30] [40,40,40,40,40,40]) # [100,100,100,100,100,100]
model_type=('Encoder_Decoder_two_models_sequential') 
project_name="6g_17th_July"

for trace in ${trace_loss[@]}; do
    for r_val in ${r[@]}; do
        for model in ${model_type[@]}; do
            for snr in ${validate_snr[@]}; do
                for train_snr in ${train_snr_dbs[@]}; do
                    for classfy_images in ${classfy_images[@]}; do
                        python ltrain/train.py decoder.use_trace_loss=$trace encoder.r=$r_val model_type=$model logger.wandb_project=$project_name reconstruct_images=$reconstruct_images validate_snr_db=$snr transmit_cls_token=$transmit_cls_token train_snr_db=$train_snr classfy_images=$classfy_images
                    done
                done
            done
        done
    done
done
