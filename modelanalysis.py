import os, sys, time
import pickle as pk
import pdb
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors

import torch, torchvision
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
import torch.multiprocessing as mlp
import torch.utils.tensorboard as tb
import torch.nn.functional as torchf


from utils import imdb
progpath = os.path.dirname(os.path.realpath(__file__))          # /home/luowei/Codes/feasc-msc
sys.path.append(progpath)
import modellearning
import myresnetmodels
# import init_normal_branch


device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
# device = torch.device('cpu')

############################################# initialize params
datasetname = "cubbirds"
image_size = 224
batchsize = 4
nthreads = 8
attention_flag = True

networkname = 'resnet18attention'
cluster = False
mymodels = myresnetmodels if networkname.find('resnet')>-1 else myvggnetmodels
modelpath = './models'

############################################## model zoo and dataset path
if cluster:
    modelzoopath = "/vulcan/scratch/cswluo/Codes/pymodels"
    datasets_path = os.path.expanduser("/vulcan/scratch/cswluo/Datasets")
else:
    modelzoopath = "/home/luowei/Codes/pymodels"
    datasets_path = os.path.expanduser("~/Datasets")
sys.path.append(os.path.realpath(modelzoopath))
# import pymodels
datasetpath = os.path.join(datasets_path, datasetname)



#################################################  organizing data
assert imdb.creatDataset(datasetpath, datasetname=datasetname) == True, "Failing to creat train/val/test sets"
data_transform = {
    'trainval': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# organizing datasets
datasplits = {x: datasets.ImageFolder(os.path.join(datasetpath, x), data_transform[x])
              for x in ['trainval', 'test']}

# preparing dataloaders for datasets
dataloader = {x: torch.utils.data.DataLoader(datasplits[x], batch_size=batchsize, shuffle=True, num_workers=nthreads)
              for x in ['trainval', 'test']}

datasplit_sizes = {x: len(datasplits[x]) for x in ['trainval', 'test']}
class_names = datasplits['trainval'].classes
num_classes = len(class_names)


############################################################################################################
############################################################################# constructing or loading model

############ number of attentions for different datasets
if datasetname in ['stcars', 'vggaircraft', 'nabirds', 'cubbirds']:
    nparts = 2
elif datasetname in ['stdogs']:
    nparts = 3
else:
    nparts = 1  # number of parts you want to use for your dataset


############ constructing models with different backbones
if networkname.find('vgg') > -1:
    pass
    # model = mymodels.vgg19_bn(pretrained=False, modelpath=modelzoopath, nparts=nparts, num_classes=num_classes,sharemask=True)
    # state_dict_path = os.path.join(modelzoopath, "vgg19_bn-c79401a0.pth")
    # pretrained_model = os.path.split(state_dict_path)[-1]
    # state_params = torch.load(state_dict_path, map_location=device)

    # model.load_state_dict(state_params, strict=False)
    # model = init_normal_branch.initVggnet(pretrained_model, state_params, model)

elif networkname.find('resnet') > -1:

   
    # resnet with attention
    model = mymodels.resnet18(pretrained=False, model_dir=modelzoopath, nparts=nparts, num_classes=num_classes, attention=attention_flag)

    modelname = r"cubbirds-resnet18attention-parts2-sps0.001-sgd-lr0.01-224-19-Mar-2020-16:23.model"
    model_state_dict = torch.load(os.path.join(modelpath, modelname))
    
    # remove 'module.' from keys
    state_dict = OrderedDict()
    for key, value in model_state_dict.items():
        new_key = key.replace('module.', '')
        state_dict[new_key] = value.cpu()
    del model_state_dict

    # initializing model using pretrained params except the modified layers
    model.load_state_dict(state_dict, strict=True)

else:
    pass



############################################################################################################
############################################################################################### show maps
# pdb.set_trace()
images, labels = next(iter(dataloader['test']))
grid = torchvision.utils.make_grid(images,50)

# pdb.set_trace()
import numpy as np
npimages = grid.numpy().transpose(1,2,0)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
npimages = std * npimages + mean
npimages = np.clip(npimages, 0, 1)
fig0 = plt.figure(num=0)
plt.imshow(npimages)
# plt.show()

# attention maps
_, _, attention_maps = model(images)    # attention_maps is with size of nbatches * nparts * height * width
nbatches, nparts, height, width = attention_maps.shape
attention_maps = torchf.interpolate(attention_maps, 224, mode='bilinear', align_corners=True)


# pdb.set_trace()
fig1, ax1 = plt.subplots(batchsize, nparts)
img_axes = []
for i in range(len(attention_maps)):
    for j in range(nparts):
        # pdb.set_trace()
        img = Image.fromarray(torch.squeeze(attention_maps[i, j]).detach().numpy())
        # img = Image.fromarray(torch.squeeze(attention_maps[i, j]).detach().numpy()).resize((image_size, image_size), resample=Image.BICUBIC)
        img = np.array(img)
        img_axes.append(ax1[i,j].imshow(img, cmap='viridis'))

# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in img_axes)
vmax = max(image.get_array().max() for image in img_axes)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in img_axes:
    im.set_norm(norm)
fig1.colorbar(img_axes[0], ax=ax1, orientation='horizontal', fraction=.1)

        
############################################################################################################
########################################################################################### show filters
attention_filters = torch.squeeze(state_dict['attention_cbns.conv1.weight'])
fig3, ax3 = plt.subplots(nparts,1)
for i in range(len(attention_filters)):
    ax3[i].stem(attention_filters[i].numpy())
plt.show()


