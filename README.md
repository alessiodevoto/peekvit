# Peekvit
Implementation of Vision Transformers & variants for experiments.
This repo uses [hydra](https://hydra.cc/) for managing experiment in a modular way.

Before testing, don't forget to `pip install -r requirements.txt`.

### How To: run a quick experiment wiht default parameters
- Edit the path to your workspace and wandb credentials in `train_config.yaml`. 
- Run `python train/train.py` 


### How To: run an experiment
- Define training parameters (model, dataset, etc..) in the hydra file `train_config.yaml`. All explanations provided inside the file.
- Run `python train/train.py` 

Notice that each time you run an experiment, a directory in your workspace will be created. Such directory contains the checkpoints for the model (if any) and the logs. 

### How To: test a model checkpoint
- Locate the experiment directory containaning your checkpoint on your local file system and edit the `load_from` field in the `test_config.yaml`. 
- Edit the parameters for testing in the `test/` hydra configs. There you can decide whether you want to add noise or set budgets to evaluate on.
- Run  `python validate/test.py`. 

Plots for each run will be stored to the run experiment directory where the checkpoint is located, i.e. `load_from`. Plots comparing different runs, if requested via the  `cumulative_plots_dir` will be stored to a new directory that you must provide as `cumulative_plots_dir` field.

### How To: write code for a new model/dataset
- Code the new model/dataset and store it somewehere in the repo.
- Add a configuration file for your model/dataset in the configuration direcotories.
- Edit the `train_config.yaml` to use your model/dataset.


#### TODOS
- fix local logging, now it has to many prints
- fix the plots for comparing multiple runs
