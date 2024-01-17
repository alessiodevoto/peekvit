import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse

from peekvit.utils.utils import make_experiment_directory, save_state, load_state
from peekvit.utils.logging import SimpleLogger, WandbLogger
from peekvit.data.dataset import Imagenette

import hydra
from hydra.utils import instantiate
import torchmetrics
from omegaconf import OmegaConf, DictConfig




@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):

    torch.manual_seed(cfg.seed)

    # experiment name and settings
    exp_name = cfg.experiment_name
    device = torch.device(cfg.device)
    experiment_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    load_from = cfg.load_from

    training_args = cfg.training
    
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoints_dir = join(experiment_dir, 'checkpoints')

    # logger
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = instantiate(cfg.logger, settings=config_dict, dir=experiment_dir)
    

    # dataset and dataloader
    dataset = instantiate(cfg.dataset)
    train_dataset, val_dataset = dataset.train_dataset, dataset.val_dataset
    train_loader = DataLoader(train_dataset, batch_size=training_args.train_batch_size, shuffle=True, num_workers=training_args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args.eval_batch_size, shuffle=False, num_workers=training_args.num_workers, pin_memory=True)

    # model
    model = instantiate(cfg.model)
    model.to(device)

    # load from checkpoint
    # we have 2 different cases:
    # 1. load from a checkpoint of the same model class
    # 2. load from a checkpoint of a different model class
    # in the first case, we can just load the state dict
    # in the second case, we need to load the state dict and do some additional operations
    print(load_from)
    if load_from is not None:
        print(cfg.model._target_, model.__class__.__name__)
        model_checkpoint, _, epoch, _, noise_args = load_state(load_from)
        print('Loading model from checkpoint of class: ', model_checkpoint.__class__.__name__)
        model.load_state_dict(model_checkpoint.state_dict(), strict=False)



    # losses 
    main_criterion = instantiate(cfg.main_criterion)

    # metrics
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.model.num_classes).to(device)

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args['lr'])

    def train_epoch(model, loader, optimizer, epoch):
        model.train()
        for batch, labels in tqdm(loader, desc=f'Training epoch {epoch}'):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(batch)
            main_loss = main_criterion(out, labels) 
            loss = main_loss 
            loss.backward()
            optimizer.step()
            
            logger.log({'train/loss': main_loss.detach().item()})


    @torch.no_grad()
    def validate_epoch(model, loader, epoch):
        model.eval()
        
        batches_loss = 0
        for batch, labels in tqdm(loader, desc=f'Validation epoch {epoch}'):
            batch, labels = batch.to(device), labels.to(device)
            out = model(batch)
            val_loss = main_criterion(out, labels) 
            _, predicted = torch.max(out, 1)
            metric(predicted, labels)
            batches_loss += val_loss.detach().item()
        
        val_loss = batches_loss / len(loader)
        acc = metric.compute()
        metric.reset()

        logger.log({'val/accuracy': acc, 'val/loss': val_loss})
        return acc, val_loss
    

    for epoch in range(training_args['num_epochs']+1):
        
        train_epoch(model, train_loader, optimizer, epoch)
        
        if training_args['eval_every'] != -1 and epoch % training_args['eval_every'] == 0:
            validate_epoch(model, val_loader, epoch)
            
        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, cfg.model, cfg.noise, optimizer, epoch)



    
if __name__ == '__main__':
    train()