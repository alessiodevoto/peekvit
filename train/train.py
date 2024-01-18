import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os.path import join
import argparse

from peekvit.utils.utils import get_last_checkpoint_path, save_state, load_state, make_experiment_directory
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
    experiment_dir, checkpoints_dir = make_experiment_directory(experiment_dir)
    
    # path to checkpoint to load from or None
    load_from = cfg.load_from

    training_args = cfg.training
    
    

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

    # load from checkpoint if requested
    if load_from is not None:
        # load from might be a path to a checkpoint or a path to an experiment directory, handle both cases
        load_from = load_from if load_from.endswith('.pth') else get_last_checkpoint_path(load_from)
        print('Loading model from checkpoint: ', load_from)
        model, _, _, _, _ = load_state(load_from, model=model)
        

    # losses 
    main_criterion = instantiate(cfg.main_criterion)

    # metrics
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.model.num_classes).to(device)

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args['lr'])

    # training loop
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

    # validation loop
    @torch.no_grad()
    def validate_epoch(model, loader, epoch):
        model.eval()
        
        batches_loss = 0
        for batch, labels in tqdm(loader, desc=f'Validation epoch {epoch}'):
            batch, labels = batch.to(device), labels.to(device)
            out = model(batch)
            val_loss = main_criterion(out, labels) 
            predicted = torch.argmax(out, 1)
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