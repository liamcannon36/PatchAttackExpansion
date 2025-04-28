import os
from parser import cfg
import PatchAttack.TextureDict_builder as TD_builder
from PatchAttack.PatchAttack_config import configure_PA
import PatchAttack.AdvPatchDict_builder as AP_builder 
from PatchAttack import utils
import torchvision.models as Models

# imageNet = '../DatasetsPatchAttack/ImageNet100'

# # os.makedirs('../DatasetsPatchAttack/ImageNet100/train/')

import urllib.request

url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
urllib.request.urlretrieve(url, 'imagenet1000_clsidx_to_labels.txt')