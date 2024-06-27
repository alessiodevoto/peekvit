from data.imagenette import Imagenette
from torch.utils.data import DataLoader
from utils.utils import load_state, get_checkpoint_path, add_noise
import torch
import torchmetrics


# change these values to what you need
device = 'cuda'
experiment_dir = '/home/aledev/projects/6G-workspace/6g_adaptive_workspace/adaptive_selection'
dataset_root = '/home/aledev/projects/6G-workspace/6g_adaptive_workspace/data'
batch_size = 64
verbose = False

torch.manual_seed(1234)

def test(snr, budget, max_eval_batches=1000):

    # download the dataset
    dataset = Imagenette(root=dataset_root, image_size=224, verbose=verbose)
    val_dataset = dataset.val_dataset
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True)


    # load the model from checkpoint
    model_checkpoints = get_checkpoint_path(experiment_dir, verbose=verbose)
    model, _, epoch, _, _ = load_state(model_checkpoints, model=None, strict=True, verbose=verbose)
    model.eval()
    model.to(device)

    # metric to evaluate the model
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=model.num_classes).to(device)

    # add a noise block, we use gaussian noise, we set the snr later
    add_noise(
        noise_type='gaussian', 
        snr=snr,
        layer=6, 
        model=model)

    # we set the snr and the budget
    snr = 1
    budget = 0.5

    model.set_budget(budget)
    


    # loop over all images in the validation set and compute average accuracy
    for i, (batch, labels) in enumerate(val_loader):
        batch, labels = batch.to(device), labels.to(device)
        out = model(batch)
        predicted = torch.argmax(out, 1)
        metric.update(predicted, labels)
        if i >= max_eval_batches:
            break
        

    acc = metric.compute()
    metric.reset()

    # comment this line if you don't want to print the accuracy
    print(f'budget: {budget}, snr: {snr}, accuracy: {acc}')

    return acc


# see here for calling this from matlab https://uk.mathworks.com/help/matlab/ref/pyrunfile.html
# it should be something like:
# acc = pyrunfile('simple_test.py', 'test', budget=<your_budget>, snr=<your_snr>)

if __name__ == '__main__':
    test(1, 0.5, 1000)

