import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet  # Use ImageNet dataset
import torch.nn as nn

from utils import *
from utils import _ntuple
import time
from ..data import ImageNet

accuracy=[]
for r in range (8,33,8):
    
    #apply the patch to the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_2tuple = _ntuple(2)
    # Load the pre-trained model
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    
    new_apply_patch(model)
    model.r = r
    source = model._tome_info["source"]
    print(model)
    
    
    # Define transformations for the ImageNet data
    # Note: The normalization values are different for ImageNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load ImageNet datahawwhatww
    imagenet_data = ImageNet(root='data', split='val', transform=transform)
    dataloader = DataLoader(imagenet_data, batch_size=64, shuffle=False)
    
    model = model.to(device)
    model.eval()
    
    # Evaluate the model
    criterion = nn.CrossEntropyLoss()
    total_val_loss = 0.0
    total_val_correct = 0
    
    start_time = time.time()  
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move inputs and labels to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
    
            # Calculate validation accuracy
            _, predicted = torch.max(outputs, 1)
            total_val_correct += (predicted == labels).sum().item()
            total_val_loss += val_loss.item()
    
    end_time = time.time()  # End time
    # Calculate average validation loss and accuracy
    avg_val_loss = total_val_loss / len(dataloader)
    val_accuracy = total_val_correct / len(dataloader.dataset)
    
    elapsed_time = end_time - start_time
    accuracy.append(val_accuracy)
    
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Elapsed Time: {elapsed_time:.2f} seconds')
print(accuracy)    