import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torchmetrics
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from pprint import pprint
from torch.utils.data import Subset



from peekvit.utils.utils import get_checkpoint_path, save_state, load_state, make_experiment_directory
from peekvit.models.topology import reinit_class_tokens, train_only_these_params
from peekvit.utils.losses import LossCompose
from peekvit.utils.visualize import plot_masked_images
from peekvit.models.topology import remove_layers_and_stitch





@hydra.main(version_base=None, config_path="../configs", config_name="train_config_personal")
def train(cfg: DictConfig):

    torch.manual_seed(cfg.seed)

    # experiment name and settings
    exp_name = cfg.experiment_name
    device = torch.device(cfg.device)
    experiment_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    experiment_dir, checkpoints_dir = make_experiment_directory(experiment_dir)
    
    
    # logger
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(config_dict)
    logger = instantiate(cfg.logger, settings=str(config_dict), dir=experiment_dir)
    

    # dataset and dataloader
    training_args = cfg.training
    dataset = instantiate(cfg.dataset)
    train_dataset, val_dataset = dataset.train_dataset, dataset.val_dataset
    train_loader = DataLoader(train_dataset, batch_size=training_args.train_batch_size, shuffle=True, num_workers=training_args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args.eval_batch_size, shuffle=False, num_workers=training_args.num_workers, pin_memory=True)

    # model
    model = instantiate(cfg.model)
    model.to(device)

    # load from checkpoint if requested
    load_from = cfg.load_from
    if load_from is not None:
        # load from might be a path to a checkpoint or a path to an experiment directory, handle both cases
        load_from = load_from if load_from.endswith('.pth') else get_checkpoint_path(load_from)
        print('Loading model from checkpoint: ', load_from)
        model, _, _, _, _ = load_state(load_from, model=model)
    
    # edit model here if requested
    if training_args['reinit_class_tokens']:
        model = reinit_class_tokens(model)
    

    if training_args['remove_layers']:
        model = remove_layers_and_stitch(model, training_args['remove_layers'])
        
    

    # main loss 
    main_criterion = instantiate(cfg.loss.classification_loss)
    
    # we might have N additional losses
    # so we store the in a dictionary
    additional_losses = None
    if cfg.loss.additional_losses is not None:
        additional_losses = LossCompose(cfg.loss.additional_losses)
        
    # metrics
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.model.num_classes).to(device)

    # optimizer and scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = None
    if 'scheduler' in cfg:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    if 'train_budget' in training_args:
        if not hasattr(model, 'set_budget'):
            print('[WARNING] Model does not have a budget attribute. Ignoring train_budget...')
        else:
            print('Setting budget to ', training_args['train_budget'])
            model.set_budget(training_args['train_budget'])
            if hasattr(model, 'enable_ranking'):
                model.enable_ranking(True)

    # training loop
    def train_epoch(model, loader, optimizer, epoch):
        model.train()
        if not training_args['train_backbone']:
            model = train_only_these_params(model, ['gate', 'class', 'head', 'threshold', 'budget'], verbose=epoch==0)
        
        """if 'train_budget' in training_args and hasattr(model, 'set_budget'):
            # train_budget = training_args['train_budget'] 
            # interpolate budget between 1 and train_budget
            budget_warmup_epochs = 5 #training_args['budget_warmup_epochs']
            step_size = (1 - training_args['train_budget'] ) / (training_args['num_epochs'] - budget_warmup_epochs)
            # train_budget = 1 - step_size * epoch
            train_budget = 1 - step_size * (epoch - budget_warmup_epochs)
            train_budget = min(train_budget, 1)
            print(f'Setting fixed training budget {train_budget:.3f} for epoch {epoch}')
            model.set_budget(train_budget)
            if hasattr(model, 'enable_ranking'):
                if train_budget < 1.0:
                    model.enable_ranking(True)
                else:
                    model.enable_ranking(False)"""


        for batch, labels in tqdm(loader, desc=f'Training epoch {epoch}'):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(batch)
            main_loss = main_criterion(out, labels) 
            add_loss_dict, add_loss_val = {}, 0.0
            if additional_losses is not None:
                add_loss_dict, add_loss_val = additional_losses.compute(
                    model, 
                    budget=getattr(model, 'current_budget', None),
                    dict_prefix='train/')
            loss = main_loss + add_loss_val
            loss.backward()
            # Apply gradient clipping
            if training_args['clip_grad_norm'] is not None:
                clip_grad_norm_(model.parameters(), max_norm=training_args['clip_grad_norm'])
            optimizer.step()
            logger.log({'train/total_loss': loss.detach().item(), 'train/classification_loss': main_loss.detach().item()} | add_loss_dict)
        
        if scheduler:
            logger.log({'train/lr': scheduler.get_last_lr()[0]})
            scheduler.step()

    @torch.no_grad()
    def validate_epoch(model, loader, epoch, budget=''):
        model.eval()
        batches_loss = 0
        for batch, labels in tqdm(loader, desc=f'Validation epoch {epoch} {budget}'):
            batch, labels = batch.to(device), labels.to(device)
            out = model(batch)
            val_loss = main_criterion(out, labels) 
            predicted = torch.argmax(out, 1)
            metric(predicted, labels)
            batches_loss += val_loss.detach().item()
        
        val_loss = batches_loss / len(loader)
        acc = metric.compute()
        metric.reset()

        return acc, val_loss

    # validation loop
    @torch.no_grad()
    def validate(model, loader, epoch):
        model.eval()
        val_budgets = cfg.training.val_budgets or [1.]
        if hasattr(model, 'set_budget'):
            for budget in val_budgets:
                model.set_budget(budget)
                acc, val_loss = validate_epoch(model, loader, epoch, budget=f'budget_{budget}')
                logger.log({f'budget_{budget}/val/accuracy': acc, f'budget_{budget}/val/loss': val_loss})
        else:
            acc, val_loss = validate_epoch(model, loader, epoch)
            logger.log({'val/accuracy': acc, 'val/loss': val_loss})
        
        return acc, val_loss


    # Aux function to plot masks during training   
    # Assumes model has budget 
    def plot_masks_in_training(model, budgets):
        
        subset_idcs = torch.arange(0, len(val_dataset), len(val_dataset)//training_args['num_images_to_plot'])
        images_to_plot = Subset(val_dataset, subset_idcs)
        hard_prefix = 'hard_'

        for budget in budgets:

            model.set_budget(budget)
            
            images = plot_masked_images(
                            model,
                            images_to_plot,
                            model_transform=None,
                            visualization_transform=dataset.denormalize_transform,
                            hard=True,
                        )
            
            os.makedirs(f'{experiment_dir}/images/epoch_{epoch}', exist_ok=True)
            os.makedirs(f'{experiment_dir}/images/epoch_{epoch}/budget_{budget}', exist_ok=True)
            for i, (_, img) in enumerate(images.items()):
                img.savefig(f'{experiment_dir}/images/epoch_{epoch}/budget_{budget}/{hard_prefix}{subset_idcs[i]}.png')
        
        
    
    # Training
    for epoch in range(training_args['num_epochs']+1):

        train_epoch(model, train_loader, optimizer, epoch)
        
        if training_args['eval_every'] != -1 and epoch % training_args['eval_every'] == 0:
            validate(model, val_loader, epoch)
            
        if training_args['checkpoint_every'] != -1 and epoch % training_args['checkpoint_every'] == 0:
            save_state(checkpoints_dir, model, cfg.model, cfg.noise, optimizer, epoch)
        
        if training_args['plot_masks_every'] != -1 and epoch % training_args['plot_masks_every'] == 0:
            if hasattr(model, 'set_budget'):
                plot_masks_in_training(model, cfg.training.val_budgets)
            else:
                print('[WARNING] Plotting masks is only supported for models with a budget. Skipping...')
            



    
if __name__ == '__main__':
    train()