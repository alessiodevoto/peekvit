# Peekvit
Implementation of Vision Transformers & variants for experiments.
This repo uses [hydra](https://hydra.cc/) for managing experiment in a modular way.

> Before testing, don't forget to `pip install -r requirements.txt`.

All settings (workspace path, dataset, model etc...) are defined in the `config` directory. With Hydra, you can edit the settings by either passing them as command line args or by editing the config files. 

Here is a list of 'HowTos' for hydra experimetns. Last one is a simple example without Hydra.

### How To: run a quick experiment with default parameters
- Edit the path to your workspace and wandb credentials in `train_config.yaml`. 
- Run `python train/train.py` 

This will download the Imagenette dataset and train a small Vision transformer on it. Notice that each time you run an experiment, a directory in your `workspace` will be created. Such directory contains the checkpoints for the model (if any) and the logs. 


### How To: write code for a new model/dataset
- Code the new model/dataset and store it somewehere in the repo.
- Add a configuration file for your model/dataset in the configuration direcotories.
- Edit the `train_config.yaml` to use your model/dataset.


### How To: run an experiment with your custom model
- Define experiment settings (model, dataset, training, losses etc..) in the hydra file `train_config.yaml`. All explanations provided inside the file.
- Run `python train/train.py` 

If your model has a sort of trainable budget, is trained starting from a checkpoint or has custom regularizations, don't forget to set the correspoding parameters in the training configurations (`train_backbone`, `reinit_class_token`, `losses` etc...) 


### How To: test a model checkpoint
- Locate the experiment directory containaning your checkpoint on your local file system and edit the `load_from` field in the `test_config.yaml`. 
- Edit the parameters for testing in the `test/` hydra configs. There you can decide whether you want to add noise or set budgets to evaluate on.
- Run  `python validate/test.py`. 

Plots for each run will be stored to the run experiment directory where the checkpoint is located, i.e. `load_from`. Plots comparing different runs, if requested via the  `cumulative_plots_dir` will be stored to a new directory that you must provide as `cumulative_plots_dir` field.


### How To: test a model checkpoint and create masking plots
- Locate the experiment directory containaning your checkpoint on your local file system and edit the `load_from` field in the `test_config.yaml`. 
- Edit the parameters for testing in the `test/` hydra configs. There you can decide how many images you want to plot.
- Run  `python validate/create_mask_plots.py`.


You will find the masking plots for the number of images you requested inside the experiment directory. 
![](images/example_plot.jpg)


### How To: Load a model in a Jupyter Notebook
- See `peekvit/notebooks/quickstart.ipynb`



### Example Scripts

In all the following scripts we pass the settings via CLI instead of editing the YAML config files.

- Train a ViT with default params: `python train/train.py`
- Train a ViT with custom params: `python train/train.py model.hidden_dim=318` 
- Fine-tune a ResidualViT starting from a vit checkpoint: `python train/train.py model=residualvit training=finetuning losses=crossentropy_mse load_from=<path_to_experiment_directory>`
- Test a model that has a budget token with different budgets `python validate/test.py load_from=<path_to_experiment_directory> 'test.budgets=[0.2, 0.4, 0.8, 1]'`
- Test a model that has a budget token with different budgets `python validate/test.py load_from=<path_to_experiment_directory> noise=gaussian noise.layer=2 'test.noises=[0.2, 0.4]'`
- Test a RankingVit initialized with the checkpoint of a ViT `python validate/test.py model=rankvit load_from=<path_to_experiment_directory>`
- Plot the class tokens and their distances for a pretrained vit_b_16 `python validate/create_cls_token_plots.py test.num_images=10 model=vit_b_16 dataset.image_size=224 model.torch_pretrained_weights=ViT_B_16_Weights['IMAGENET1K_V1']`
- Run evaluation on a pretrained vision transformer on imagenet `CUDA_VISIBLE_DEVICES=3 python validate/test.py noise=no_noise  model=vit_b_16_pretrained dataset=imagenet test.test_batch_size=2048`



#### TODOS
- Detailed comments for losses
- Examples on comparative plots
- Fix autoenc loss for masking 
- Class token plot example
- Add a script for each howto
- default cumulative noise plot when testing noise
- fix enc dec as now computes loss for all tokens
- config file for each model
- change cuda float
- change noise in config
