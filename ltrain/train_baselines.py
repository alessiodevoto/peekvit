import os, sys
import json
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
from timm_models_baselines import *
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
    # load_from = cfg.load_from
    # if load_from is not None:
    #     # load from might be a path to a checkpoint or a path to an experiment directory, handle both cases
    #     load_from = (
    #         load_from if load_from.endswith(".pth") else get_checkpoint_path(load_from)
    #     )
    #     print("Loading model from checkpoint: ", load_from)
    #     model, _, _, _, _ = load_state(load_from, model=model)

    # # edit model here if requested
    # if training_args["reinit_class_tokens"]:
    #     model = reinit_class_tokens(model)

    # Main loss
    #main_criterion = instantiate(cfg.loss.classification_loss)

    # we might have N additional losses
    # so we store the in a dictionary
    # additional_losses = None
    # if cfg.loss.additional_losses is not None:
    #     additional_losses = LossCompose(cfg.loss.additional_losses)

    # # metrics
    # metric_mse = torchmetrics.MeanMetric().to(device)
    # metric_cl_loss = torchmetrics.MeanMetric().to(device)
    
    metric_acc = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=cfg.encoder.num_classes
    ).to(device)

    # Optimizer and scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = None
    if "scheduler" in cfg:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    def plot_reconstructed_images_in_training(model, epoch, snr_db):
        if epoch == -1:
            epoch = "best"

        subset_idcs = torch.arange(
            0, len(val_dataset), len(val_dataset) // training_args["num_images_to_plot"]
        )
        images_to_plot = Subset(val_dataset, subset_idcs)
        

        images = plot_reconstructed_images(
            model,
            images_to_plot,
            model_transform=None,
            visualization_transform=dataset.denormalize_transform,
            snr_db=snr_db
        )

        os.makedirs(f"{experiment_dir}/images/epoch_{epoch}", exist_ok=True)
        os.makedirs(
            f"{experiment_dir}/images/epoch_{epoch}/reconstructed/{snr_db}",
            exist_ok=True,
        )
        for i, (_, img) in enumerate(images.items()):
            img.savefig(
                f"{experiment_dir}/images/epoch_{epoch}/reconstructed/{snr_db}/reconstructed_img_{subset_idcs[i]}.png"
            )
    
    def plot_reconstructed_images_pergroup_in_training(model, epoch, snr_db):
        if epoch == -1:
            epoch = "best"

        subset_idcs = torch.arange(
            0, len(val_dataset), len(val_dataset) // training_args["num_images_to_plot"]
        )
        images_to_plot = Subset(val_dataset, subset_idcs)
        

        images = plot_reconstructed_images_pergroup(
            model,
            images_to_plot,
            model_transform=None,
            visualization_transform=dataset.denormalize_transform,
            snr_db=snr_db
        )

        os.makedirs(f"{experiment_dir}/images/epoch_{epoch}", exist_ok=True)
        os.makedirs(
            f"{experiment_dir}/images/epoch_{epoch}/reconstructed/{snr_db}/pergroup",
            exist_ok=True,
        )
        for i, (_, img) in enumerate(images.items()):
            img.savefig(
                f"{experiment_dir}/images/epoch_{epoch}/reconstructed/{snr_db}/pergroup/image_{subset_idcs[i]}.png"
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

        for batch, labels in tqdm(loader, desc=f"Training epoch {epoch}"):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            model_out = model(batch, labels, snr_db=snr_db)
            
            reconstruction_loss = model_out["reconstruction_loss"]
            classification_loss = model_out["classification_loss"]
 

            loss = reconstruction_loss + classification_loss
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
                }
                # | add_loss_dict
            )

        if scheduler:
            logger.log({"train/lr": scheduler.get_last_lr()[0]})
            scheduler.step()

    @torch.no_grad()
    def validate_epoch(model, loader, epoch, snr_db=0):
        model.eval()
        batches_loss_mse, batches_loss_cl = 0, 0
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
            metric_acc(val_class_preds, labels)        
            batches_loss_mse += val_reconstruction_loss.detach().item()
            batches_loss_cl += val_classification_loss.detach().item()

        # Cost
        val_loss_mse = batches_loss_mse / len(loader)
        val_loss_cl = batches_loss_cl / len(loader)
    
        # Acc metric
        acc = metric_acc.compute()
        metric_acc.reset()
    

        return acc, val_loss_mse, val_loss_cl

    # validation loop
    @torch.no_grad()
    def validate(model, loader, epoch, snr_db=5, log=True):
        model.eval()
        
        acc, val_loss_mse, val_loss_cl = validate_epoch(model, loader, epoch, snr_db=snr_db)
        if log==True:
            logger.log({"val/mse": val_loss_mse, 
                        "val/classification_loss": val_loss_cl,
                        "val/accuracy": acc,
                        "val/loss": val_loss_mse + val_loss_cl})

        return acc, val_loss_mse, val_loss_cl

    train_snr_bd = None if cfg.train_snr_db == "random" else cfg.train_snr_db 
    
    
    # Training
    for epoch in range(1, training_args["num_epochs"] + 1):
        #plot_reconstructed_images_in_training(model, epoch=-1, snr_db=snr_db)
        # Plot per group
        
        train_epoch(model, train_loader, optimizer, epoch, train_snr_bd)
        

        if (
            training_args["eval_every"] != -1
            and epoch % training_args["eval_every"] == 0
        ):
            validation_acc, val_loss_mse, val_loss_cl = validate(model, val_loader, epoch, snr_db=100)
            
            if validation_acc > model.best_validation_acc:
                model.best_validation_acc = validation_acc
                model.best_val_loss_mse = val_loss_mse
                model.best_val_loss_cl = val_loss_cl
                save_state(checkpoints_dir, model, None, None, optimizer, epoch)

            # plot_reconstructed_images_in_training(model)

        
    logger.log({"val/best_mse": model.best_val_loss_mse, 
                "val/best_classification_loss": model.best_val_loss_cl,
                "val/best_accuracy": model.best_validation_acc,
                "val/best_loss": model.best_validation_acc + model.best_val_loss_cl})
    
    # Evaluation part:
    # 1st load the best model:
    path_to_run = '/'.join(checkpoints_dir.split('/')[:-1])
    best_model_path = get_checkpoint_path(path_to_run)
    model, _, _, _, _ = load_state(best_model_path, model=model)
    results_collector = {}
    for snr_db in range(-10, 11, 1):
        test_acc, test_loss_mse, test_loss_cl = validate(
            model,
            val_loader,
            epoch=-1,
            snr_db=snr_db,
            log=False
        )
        
        logger.log({"test/mse": test_loss_mse, 
                    "test/classification_loss": test_loss_cl,
                    "test/accuracy": test_acc,
                    "test/loss": test_loss_mse + test_loss_cl})

        results_collector[snr_db] = {
            "test/mse": test_loss_mse, 
            "test/classification_loss": test_loss_cl,
            "test/accuracy": test_acc,
            "test/loss": test_loss_mse + test_loss_cl
        }

        # plot_reconstructed_images_in_training(model, epoch=-1, snr_db=snr_db)
        # Plot per group
    if cfg.plot_groups == True:
        plot_reconstructed_images_pergroup_in_training(model, epoch=-1, snr_db=0)

            
    # Save the collected results into path_to_run
    # as a json file
    
    # Make sure to convert all tensor values to python scalars
    for key in results_collector.keys():
        results_collector[key] = {k: v.item() if hasattr(v, "item") else v for k, v in results_collector[key].items()}
        

    with open(f"{path_to_run}/results.json", "w") as f:
        json.dump(results_collector, f)
    logger.close()
    # Delete all unnecessary checkpoints paths keep only best_model_path
    
    # Delete all checkpoints except the best one
    checkpoint_dir = os.path.join(path_to_run, "checkpoints")
    for file in os.listdir(checkpoint_dir):
        #if file.endswith(".pth") and file != best_model_path.split('/')[-1]:
        os.remove(os.path.join(checkpoint_dir, file))




import matplotlib.pyplot as plt
def plot_reconstructed_images(
    model, 
    images_to_plot,
    model_transform, 
    visualization_transform,
    snr_db,
):
    figs = {}
    i = 0
    for img, label in tqdm(images_to_plot, desc="Preparing reconstructed images plots"):

        # Forward pass
        _img = model_transform(img) if model_transform is not None else img
        device = model.decoder_pred.weight.device
        model_out = model(make_batch(_img).to(device),
                        torch.tensor([label]).long().to(device),
                        return_pred_images=True, snr_db=snr_db)
        
        reconstructed = model_out["reconstructed_image"]


        # Prepare plot, we want a row for each residual layer,
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
        source = model.mae_encoder._tome_info["source"] if hasattr(model.mae_encoder, "_tome_info") else None
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
    vis_img = img

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        # vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        # vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)
        # Adjust the brightness of the image where the mask is applied
        vis_img = vis_img * (1 - mask_eroded) + mask_eroded * (0.8 * vis_img + 0.2 * color)

        # Blend with the colormap
        vis_img += mask_edge * np.array(cmap[i]).reshape(1, 1, 3)
    
    vis_img = np.clip(vis_img, 0, 1)
    # Convert back into a PIL image
    # vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img

from scipy.ndimage import binary_erosion, binary_dilation
# def make_mask_visualization_pergroup(
#     img, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
# ):
#     """
#     Create a visualization with thicker, more colorful, and distinct borders.

#     Args:
#         img (np.array): Original image as a numpy array (RGB format).
#         source (torch.Tensor): Tensor containing segmentation/classification information.
#         patch_size (int): Size of patches for segmentation.
#         class_token (bool): Whether to include a class token.

#     Returns:
#         np.array: Modified image with masks overlayed.
#     """

#     source = source.detach().cpu()

#     h, w, _ = img.shape
#     ph = h // patch_size
#     pw = w // patch_size

#     if class_token:
#         source = source[:, :, 1:]

#     vis = source.argmax(dim=1)
#     num_groups = vis.max().item() + 1

#     cmap = tome.vis.generate_colormap(num_groups)
#     vis_img = img.copy()

#     for i in range(num_groups):
#         mask = (vis == i).float().view(1, 1, ph, pw)
#         mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
#         mask = mask.view(h, w, 1).numpy()

#         # Calculate the color for the current mask
#         color = (mask * img).sum(axis=(0, 1)) / (mask.sum() + 1e-8)  # Avoid division by zero

#         # Erode the mask to create the inner part and edges
#         mask_eroded = binary_erosion(mask[..., 0], iterations=2)[..., None].astype(float)

#         # Calculate the edges by subtracting the eroded mask from the original mask
#         mask_edge = mask - mask_eroded

#         # Make the edges more rigid and distinct by intensifying the color
#         vis_img = vis_img * (1 - mask_edge) + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

#         # Optionally, darken the original mask to make the edges stand out more
#         vis_img = vis_img * (1 - mask_eroded) + mask_eroded * (0.5 * vis_img + 0.5 * color)

#     # Ensure image values are within the valid range
#     vis_img = np.clip(vis_img, 0, 1)

#     return vis_img


def make_mask_visualization_pergroup(
    img, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
):
    """
    Create a visualization with colored borders for groups provided in the source,
    with the last group specifically having a black border.

    Args:
        img (np.array): Original image as a numpy array (RGB format).
        source (torch.Tensor): Tensor containing segmentation/classification information.
        patch_size (int): Size of patches for segmentation.
        class_token (bool): Whether to include a class token.

    Returns:
        np.array: Image with overlayed borders.
    """

    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() 

    # Colors for each group, specify black for the last group
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [0.5, 0, 0.5],  # Violet
        [1, 0.843, 0],  # Pink
    ]

    colors[:num_groups] +  [[0, 0, 0]] # Black for the last group

    vis_img = img.copy()  # Start with the original image

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w).numpy()

        # Erode the mask to create the inner part and edges
        mask_eroded = binary_erosion(mask, iterations=2)
        mask_edge = mask - mask_eroded

        # Color the edges using predefined colors
        edge_color = np.array(colors[i], dtype=np.float32)
        for c in range(3):  # Apply color to the edge
            vis_img[:, :, c] = np.where(mask_edge, edge_color[c], vis_img[:, :, c])

    # Ensure image values are within the valid range
    vis_img = np.clip(vis_img, 0, 1)

    return vis_img




def plot_reconstructed_images_pergroup(
    model, 
    images_to_plot,
    model_transform, 
    visualization_transform,
    snr_db,
):
    figs = {}
    i = 0
    
    for img, label in tqdm(images_to_plot, desc="Preparing reconstructed images plots"):
        
        # Create a new figure for each image
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Remove white borders
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.margins(0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        # Forward pass
        _img = model_transform(img) if model_transform is not None else img
        device = model.decoder_pred.weight.device
        model_out = model(make_batch(_img).to(device),
                        torch.tensor([label]).long().to(device),
                        return_pred_images=True, snr_db=snr_db)

        # Prepare the original image for plotting
        img = prepare_for_matplotlib(
            visualization_transform(img) if visualization_transform is not None else img
        )

       
       
        # Visualizations with mask
        source = model.mae_encoder._tome_info["source"] if hasattr(model.mae_encoder, "_tome_info") else None
        vals, indices = torch.topk(source.sum(-1).flatten(), k=5)

        # Generate the mask visualization
        mask_img = make_mask_visualization_pergroup(
            img, 
            #source[:, indices, :],
            torch.concat([source[:, indices, :], (1 - source[:, indices, :].sum(1)).unsqueeze(1)], dim=1), 
            class_token=model.mae_encoder.cls_token is not None
        ) 
        
        # Plot the mask image
        ax.imshow(mask_img)
        
        # Store the figure in the dictionary with a unique key
        figs[f"reconstructed_{i}"] = fig
        
        i += 1
    return figs







if __name__ == "__main__":
    train()
