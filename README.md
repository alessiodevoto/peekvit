# peekvit

Fast prototyping for Vision Transformers and MoEs + visualizations.

- Open `train_moe.py`, edit the hyperparameters and paths at the beginning of the file, and then run `python train_moe.py` without any args. 
- This will run a training and compute visualizations on the provided subset of the validation set. Model checkpoints and images will be saved into the provided `BASE_PATH` variable. 
- In case you only want to see visualization without rerunning the training, edit the path to `run_dir` and comment the `train()` line.
