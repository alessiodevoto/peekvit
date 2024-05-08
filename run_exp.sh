python ltrain/train_enc_dec_classifier.py decoder.use_trace_loss=False experiment_name=enc_dec_classifier_no_trace_loss_r10x8 encoder.r=[10,10,10,10,10,10,10,10]
python ltrain/train_enc_dec_classifier.py decoder.use_trace_loss=True experiment_name=enc_dec_classifier_trace_loss_r10x8 encoder.r=[10,10,10,10,10,10,10,10]

python ltrain/train_classification.py experiment_name=solo_classifier_r10x8 encoder.r=[10,10,10,10,10,10,10,10]

python ltrain/train_enc_dec_classifier_onemodel.py decoder.use_trace_loss=False experiment_name=enc_dec_classifier_onemodel_no_trace_loss_r8x8 encoder.r=[8,8,8,8,8,8,8,8]
python ltrain/train_enc_dec_classifier_onemodel.py decoder.use_trace_loss=True experiment_name=enc_dec_classifier_trace_loss_onemodel_r8x8 encoder.r=[8,8,8,8,8,8,8,8]

python ltrain/train_enc_dec_classifier_onemodel.py decoder.use_trace_loss=False experiment_name=enc_dec_classifier_onemodel_no_trace_loss_r10x8 encoder.r=[10,10,10,10,10,10,10,10]
python ltrain/train_enc_dec_classifier_onemodel.py decoder.use_trace_loss=True experiment_name=enc_dec_classifier_trace_loss_onemodel_r10x8 encoder.r=[10,10,10,10,10,10,10,10]

