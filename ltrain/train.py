import os, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
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


from peekvit.utils.utils import (
    get_checkpoint_path,
    save_state,
    load_state,
    make_experiment_directory,
)
from peekvit.models.topology import reinit_class_tokens, train_only_these_params
from peekvit.utils.losses import LossCompose

from peekvit.utils.visualize import *
import timm 
import tome
from timm_models import *
import torch

@hydra.main(
    version_base=None, config_path="../configs", config_name="train_config_personal"
)

def train(cfg: DictConfig):
    torch.set_num_threads(1)
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.train_batch_size,
        shuffle=True,
        num_workers=training_args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_args.eval_batch_size,
        shuffle=False,
        num_workers=training_args.num_workers,
        pin_memory=True,
    )

    # Model
    model = MAEVisionTransformer(cfg) #instantiate(cfg.model) # timm.create_model('vit_base_patch16_224', pretrained=True) #
    model.to(device)
    model.logger = logger

    # load from checkpoint if requested
    load_from = cfg.load_from
    if load_from is not None:
        # load from might be a path to a checkpoint or a path to an experiment directory, handle both cases
        load_from = (
            load_from if load_from.endswith(".pth") else get_checkpoint_path(load_from)
        )
        print("Loading model from checkpoint: ", load_from)
        model, _, _, _, _ = load_state(load_from, model=model)

    # edit model here if requested
    if training_args["reinit_class_tokens"]:
        model = reinit_class_tokens(model)

    # Main loss
    #main_criterion = instantiate(cfg.loss.classification_loss)

    # we might have N additional losses
    # so we store the in a dictionary
    additional_losses = None
    if cfg.loss.additional_losses is not None:
        additional_losses = LossCompose(cfg.loss.additional_losses)

    # metrics
    metric_mse = torchmetrics.MeanMetric().to(device)
    metric_cl_loss = torchmetrics.MeanMetric().to(device)
    metric_acc = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=cfg.encoder.num_classes
    ).to(device)
    metric_acc_topology = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=cfg.encoder.num_classes
    ).to(device)

    # optimizer and scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = None
    if "scheduler" in cfg:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    def plot_reconstructed_images_in_training(model,):

        subset_idcs = torch.arange(
            0, len(val_dataset), len(val_dataset) // training_args["num_images_to_plot"]
        )
        images_to_plot = Subset(val_dataset, subset_idcs)
        

        images = plot_reconstructed_images(
            model,
            images_to_plot,
            model_transform=None,
            visualization_transform=dataset.denormalize_transform,
        )

        os.makedirs(f"{experiment_dir}/images/epoch_{epoch}", exist_ok=True)
        os.makedirs(
            f"{experiment_dir}/images/epoch_{epoch}/reconstructed",
            exist_ok=True,
        )
        for i, (_, img) in enumerate(images.items()):
            img.savefig(
                f"{experiment_dir}/images/epoch_{epoch}/reconstructed/reconstructed_img_{subset_idcs[i]}.png"
            )

    # training loop
    def train_epoch(model, loader, optimizer, epoch, snr_db):
        model.train()
        if not training_args["train_backbone"]:
            model = train_only_these_params(
                model,
                ["gate", "class", "head", "threshold", "budget"],
                verbose=epoch == 0,
            )

        # if "train_budget" in training_args:
        #     print("Setting budget to ", training_args["train_budget"])
        #     model.set_budget(training_args["train_budget"])
        #     if hasattr(model, "enable_ranking"):
        #         model.enable_ranking(True)

        for batch, labels in tqdm(loader, desc=f"Training epoch {epoch}"):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            model_out = model(batch, labels, snr_db=snr_db)
            
            reconstruction_loss = model_out["reconstruction_loss"]
            classification_loss = model_out["classification_loss"]
            topology_loss = model_out["topology_loss"]

            loss = reconstruction_loss + classification_loss + topology_loss
            loss.backward()

            # Apply gradient clipping
            if training_args["clip_grad_norm"] is not None:
                clip_grad_norm_(
                    model.parameters(), max_norm=training_args["clip_grad_norm"]
                )
            optimizer.step()
            logger.log(
                {
                    "train/total_loss": loss.detach().item(),
                    "train/mse_loss": reconstruction_loss.detach().item(),
                    "train/classification_loss": classification_loss.detach().item(),
                    "train/topology_loss": topology_loss.detach().item(),
                }
                # | add_loss_dict
            )

        if scheduler:
            logger.log({"train/lr": scheduler.get_last_lr()[0]})
            scheduler.step()

    @torch.no_grad()
    def validate_epoch(model, loader, epoch, snr_db=0):
        model.eval()
        batches_loss_mse, batches_loss_cl, batches_topology_loss_cl = 0, 0, 0
        for batch, labels in tqdm(loader, desc=f"Validation epoch {epoch}"):
            batch, labels = batch.to(device), labels.to(device)
            model_out = model(
                batch, 
                labels,
                return_pred_labels=True,
                snr_db=snr_db
            )
            val_reconstruction_loss = model_out["reconstruction_loss"]
            val_classification_loss = model_out["classification_loss"]
            val_class_preds = model_out["class_preds"]

            val_topology_cnn_loss = model_out["topology_loss"]
            val_topology_class_preds = model_out["topology_class_preds"]

            
            metric_acc(val_class_preds, labels)
            metric_acc_topology(val_topology_class_preds, labels)
            
            batches_loss_mse += val_reconstruction_loss.detach().item()
            batches_loss_cl += val_classification_loss.detach().item()
            batches_topology_loss_cl += val_topology_cnn_loss.detach().item()

        # Cost
        val_loss_mse = batches_loss_mse / len(loader)
        val_loss_cl = batches_loss_cl / len(loader)
        val_loss_topology_cl = batches_loss_cl / len(loader)
        
        # Acc metric
        acc = metric_acc.compute()
        acc_topology = metric_acc_topology.compute()

        metric_acc.reset()
        metric_acc_topology.reset()

        return acc, acc_topology, val_loss_mse, val_loss_cl, val_loss_topology_cl

    # validation loop
    @torch.no_grad()
    def validate(model, loader, epoch, snr_db=5):
        model.eval()
        
        acc, acc_topology, val_loss_mse, val_loss_cl, val_loss_topology_cl = validate_epoch(model, loader, epoch, snr_db=snr_db)
        logger.log({"val/mse": val_loss_mse, 
                    "val/classification_loss": val_loss_cl,
                    "val/accuracy": acc,
                    "val/loss": val_loss_mse + val_loss_cl + val_loss_topology_cl,
                     "val/topology_loss": val_loss_topology_cl,
                     "val/topology_accuracy": acc_topology,})

        return acc, acc_topology, val_loss_mse, val_loss_cl, val_loss_topology_cl

    train_snr_bd = None if cfg.train_snr_db == "random" else cfg.train_snr_db 
    # Training
    for epoch in range(training_args["num_epochs"] + 1):
        
        train_epoch(model, train_loader, optimizer, epoch, train_snr_bd)
        

        if (
            training_args["eval_every"] != -1
            and epoch % training_args["eval_every"] == 0
        ):
            validation_acc, validation_acc_topology, val_loss_mse, val_loss_cl, val_loss_topology_cl = validate(model, val_loader, epoch, snr_db=cfg.validate_snr_db)
            
            if validation_acc > model.best_validation_acc:
                model.best_validation_acc = validation_acc
                model.best_val_loss_mse = val_loss_mse
                model.best_val_loss_cl = val_loss_cl

                model.best_val_loss_topology_cl = val_loss_topology_cl
                model.best_validation_acc_topology = validation_acc_topology

                # save_state(checkpoints_dir, model, cfg.model,cfg.noise, optimizer, epoch)

            plot_reconstructed_images_in_training(model)

        
    logger.log({"val/best_mse": model.best_val_loss_mse, 
                "val/best_classification_loss": model.best_val_loss_cl,
                "val/best_topology_accuracy": model.best_validation_acc_topology,
                "val/best_accuracy": model.best_validation_acc,
                "val/best_topology_loss": model.best_val_loss_topology_cl,
                "val/best_loss": model.best_validation_acc + model.best_val_loss_cl + model.best_val_loss_topology_cl})


import matplotlib.pyplot as plt
def plot_reconstructed_images(
    model, images_to_plot, model_transform, visualization_transform
):
    figs = {}
    i = 0
    for img, label in tqdm(images_to_plot, desc="Preparing reconstructed images plots"):

        # forward pass
        _img = model_transform(img) if model_transform is not None else img

        # model.set_budget(budget)
        device = model.decoder_pred.weight.device
        #_,_, reconstructed = model(make_batch(_img).to(device), torch.tensor([label]).long().to(device), return_pred_images=True)
        model_out = model(make_batch(_img).to(device), torch.tensor([label]).long().to(device), return_pred_images=True)
        
        reconstructed = model_out["reconstructed_image"]


        # prepare plot, we want a row for each residual layer,
        # and two columns, one for the image and one for token masks
        fig, axs = plt.subplots(3, 1, squeeze=False, figsize=(10, 25))

        # plot the image
        img = prepare_for_matplotlib(
            visualization_transform(img) if visualization_transform is not None else img
        )
        axs[0, 0].imshow(img)
        axs[0, 0].title.set_text("Original image")

        # plot the reconstructed image
        reconstructed = prepare_for_matplotlib(
            visualization_transform(reconstructed).squeeze()
        )
        axs[1, 0].imshow(reconstructed)
        axs[1, 0].title.set_text("Reconstructed image")
        
        # Visualizations with mask
        source = model.mae_encoder._tome_info["source"]
        if source is not None:
            mask_img = make_mask_visualization(img, source, class_token=model.mae_encoder.cls_token is not None) 
            tokens_at_the_end = source.shape[1]
        else:
            mask_img = np.zeros_like(img)
            tokens_at_the_end = "No compression"

        axs[2, 0].imshow(mask_img)
        axs[2, 0].title.set_text(f"Mask ({tokens_at_the_end} tokens at the end)")

        fig.tight_layout()

        figs[f"reconstructed_{i}"] = fig
        i += 1

    return figs


from scipy.ndimage import binary_erosion
import numpy as np
def make_mask_visualization(
    img, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
):
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    #img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = tome.vis.generate_colormap(num_groups)
    vis_img = 0

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    # vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img

if __name__ == "__main__":
    train()
