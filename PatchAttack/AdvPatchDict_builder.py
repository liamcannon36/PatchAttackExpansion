import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as Models
import torch.optim as optim
from torch.utils.data import DataLoader
import PatchAttack.utils as utils
from PatchAttack.TextureDict_extractor import vgg19_extractor as gen_kit
import kornia
import time
from PatchAttack.PatchAttack_config import PA_cfg

torch_cuda = 0

# Custom data agent which allows for managing the dataset and dataloaders' batch size
class custom_data_agent():

    def __init__(self, dataset):
        self.train_dataset = dataset

    def update_loaders(self, batch_size):
        # Updates the dataloader with the specified batch size
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
        )
        print('Your dataloader has been updated with batch size {}'.format(self.batch_size))


#Generates a batch of training images after applying the adv. patch. Optimizes the adv. patches, and transoforming (translate, rotate, etc...) then this overlays them onto the actual image
def make_training_batch(input_tensor, patch, patch_mask):
    
    # Determine the patch size and area allowed for translation 
    H, W = PA_cfg.image_shape[-2:]
    PATCH_SIZE = int(np.floor(np.sqrt((H*W*PA_cfg.percentage))))
    translate_space = [H-PATCH_SIZE+1, W-PATCH_SIZE+1]
    bs = input_tensor.size(0)

    training_batch = []
    for b in range(bs):
        # Apply random translation and rotation to the adv. patch
        u_t = np.random.randint(low=0, high=translate_space[0])
        v_t = np.random.randint(low=0, high=translate_space[1])
        scale = np.random.rand() * (PA_cfg.scale_max - PA_cfg.scale_min) + PA_cfg.scale_min
        angle = np.random.rand() * (PA_cfg.rotate_max - PA_cfg.rotate_min) + PA_cfg.rotate_min
        center = torch.Tensor([u_t+PATCH_SIZE/2, v_t+PATCH_SIZE/2]).unsqueeze(0)
        rotation_m = kornia.get_rotation_matrix2d(center, angle, scale)

        # Warp the mask and patch using the transformations, gets the batch dimension from patch_mask
        temp_mask = patch_mask.unsqueeze(0)
        temp_input = input_tensor[b].unsqueeze(0)
        temp_patch = patch.unsqueeze(0)

        # Applies translation        
        temp_mask = kornia.translate(temp_mask.float(), translation=torch.Tensor([u_t, v_t]).unsqueeze(0))
        temp_patch = kornia.translate(temp_patch.float(), translation=torch.Tensor([u_t, v_t]).unsqueeze(0))

        # Applies rotation         
        mask_warpped = kornia.warp_affine(temp_mask.float(), rotation_m, temp_mask.size()[-2:])
        patch_warpped = kornia.warp_affine(temp_patch.float(), rotation_m, temp_patch.size()[-2:])

        # Overlay the patch onto the input image
        overlay = temp_input * (1 - mask_warpped) + patch_warpped * mask_warpped
        training_batch.append(overlay)
    
    # Concatenate the batch and return
    training_batch = torch.cat(training_batch, dim=0)
    return training_batch


# Trains models with adv patches
def build(model, t_labels, DA):
    # Determine patch size and make necessary configurations
    H, W = PA_cfg.image_shape[-2:]
    PATCH_SIZE = int(np.floor(np.sqrt((H*W*PA_cfg.percentage))))
    DA.update_loaders(PA_cfg.batch_size)

    # Initialize the patch and patch mask
    patch = torch.zeros(PA_cfg.image_shape)
    patch_mask = torch.zeros(1, H, W)
    for i in range(PATCH_SIZE):
        for j in range(PATCH_SIZE):
            patch_mask[:, i, j] = 1.

    patch = patch.requires_grad_()  # Make the patch optimizable via the model
    optimizer = optim.SGD(params=[patch], lr=PA_cfg.AP_lr)  
    criterion = nn.CrossEntropyLoss().cuda(torch_cuda)  # Loss function of our model

    #Goes through each label in our image dataset (t_label= target label)
    for AP_index in range(len(PA_cfg.AdvPatch_dirs)):
        t_label = t_labels[AP_index]

        #If patch already created, do nothing
        if os.path.exists(os.path.join(PA_cfg.AdvPatch_dirs[AP_index], 'patch_with_mask.pt')):
            print('Patch of t_label {} is already generated!'.format(t_label))
        else:
            # Train a patch using that label
            target_tensor = torch.ones(PA_cfg.batch_size).long().cuda(torch_cuda)*t_label
            time_start = time.time()

            for i, (input_tensor, label_tensor) in enumerate(DA.train_loader):
                # Create one batch of adv. patches
                training_batch = make_training_batch(input_tensor, patch, patch_mask)
                training_batch, label_tensor = training_batch.cuda(torch_cuda), label_tensor.cuda(torch_cuda)

                # Forward pass
                output = model(training_batch) 
                # Calculate loss
                loss = criterion(output, target_tensor)  
                # Backprop.
                loss.backward()  
                #Calculate accuracy
                target_acc = utils.accuracy(output.data, target_tensor, topk=(1,))
                # Update adv. patch
                optimizer.step()  
                optimizer.zero_grad()

                
                gen_kit.normalize(patch)

                print('Target: {} | Iter [{}] | Loss: {:.8f} | Target Acc: {:.4f}'.format(t_label, i, loss.item(), target_acc[0].item()))

                # Early stopping 
                if i > PA_cfg.iterations:  
                    break

            # Save the patch
            if not os.path.exists(PA_cfg.AdvPatch_dirs[AP_index]):
                os.makedirs(PA_cfg.AdvPatch_dirs[AP_index])
            torch.save((patch.detach(), patch_mask.clone()), os.path.join(PA_cfg.AdvPatch_dirs[AP_index], 'patch_with_mask.pt'))

            time_end = time.time()
            print('t_label {} finished | Time used: {}'.format(t_label, time_end - time_start))
