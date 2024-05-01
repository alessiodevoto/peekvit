# ViT + Merging + Noise channel

Code for testing [timm models](https://huggingface.co/timm) with merging and noise channel in the middle of transformer. For this experiment it is used the 
[vit_base_patch16_224](https://huggingface.co/google/vit-base-patch16-224)


## Dependencies
- python >= 3.8
- pytorch >= 1.12.1
- torchvision
- timm >= 0.4.12

## Dataset
Imagenet1k validation set (6.7 GB)

## How to use

You can test the accuracy of the pretrained ViT model by selecting the number of token you want to merge in each leayer (model.r). There are two functions to apply the 
patch to the existing pre-trained model:
  - **new_apply_patch(model)**: it applies the merging to the first six blocks of the transformer, then after the sixth block, it adds Gaussian noise and leaves untouched the remaining six layers
  - **apply_patch(model)**: it applies the merging to all blocks without inserting any communication channel

You can define two different types of communication channel functions:
  - **white_noise_norm(tensor,snr_lin):** it normalizes the latent space in order to get 0 mean and unit standard deviation in order to add noise with unit variance to all different tensors.
  - **white_noise_tens(tensor,snr_lin):** it computes the signal power for each tensor and computes the variance and std, then applies white noise based on the std
To change the type of noise you have to modify the **ToMeBlockMLP** class in the **utils.py**

Let's clarify the different classes in the **utils.py**:
  - **ToMeBlock**: this is the standard class from the timm repository, it is the basic pytorch transformer with the merging procedure
  - **ToMeBlocknoMerge**: same as before, it does not have the merging procedure in it
  - **ToMeBlockMLP**: this is equal to ToMeBlock but it adds the noise at the end of the MLP outout
  - **ToMeAttention**: applies proportional attention and returns the mean of k over heads from attention

    
