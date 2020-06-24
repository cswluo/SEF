import os, sys, time
import pickle as pk
import pdb
import uuid
import argparse

import torch, torchvision
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
import torch.multiprocessing as mlp
import torch.utils.tensorboard as tb


from utils import imdb
progpath = os.path.dirname(os.path.realpath(__file__))      
sys.path.append(progpath)
import modellearning
import sef

device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
device_name = device.type+':'+str(device.index) if device.type=='cuda' else 'cpu'


########################################################################################################## initialize params
datasetname = "cubbirds"
image_size = 448
batchsize = 32
nthreads = 8
lr = 1e-2
lmgm = 1
entropy = 1
soft = 0.05
epochs = 50
optmeth = 'sgd'
regmeth = 'cms'

# number of attentions for different datasets
if datasetname in ['cubbirds']:
    nparts = 4
elif datasetname in ['vggaircraft']:
    nparts = 3
elif datasetname in ['stdogs', 'stcars']:
    nparts = 2
else:
    nparts = 1  # number of parts you want to use for your dataset


# 'resnet50attention' for sef, 'resnet50maxent' for resnet with MaxEnt, 'resnet50vanilla' for the vanilla resnet
networkname = 'resnet50attention'
if networkname.find('attention') > -1:  # sef based on resnet
    attention_flag = True
    maxent_flag = False
elif networkname.find('maxent') > -1:   # resnet with the maximum entropy regularization
    lmgm=soft=0
    nparts=1
    entropy=1
    attention_flag = False
    maxent_flag = True
else:                                   # the vanilla resnet
    lmgm=entropy=soft=0
    nparts=1
    attention_flag = False
    maxent_flag = False




############################################################################################################## displaying logs
timeflag = time.strftime("%d-%b-%Y-%H:%M")
# writer = tb.SummaryWriter(log_dir='./runs/'+datasetname+'/'+networkname+time.strftime("%d-%b-%Y"))
log_items = r'{}-net{}-att{}-lmgm{}-entropy{}-soft{}-lr{}-imgsz{}-bsz{}'.format(
    datasetname, int(networkname[6:8]), nparts, lmgm, entropy, soft, lr, image_size, batchsize)
writer = tb.SummaryWriter(comment='-'+log_items)
logfile = open('./results/'+log_items+'.txt', 'w')
modelname = log_items + '.model'



############################################################################################################## model zoo and dataset path
datapath = '/path/to/your/dataset'
modelzoopath = '/path/to/the/vanilla/resnet/models'
sys.path.append(modelzoopath)
datasetpath = os.path.join(datapath, datasetname)
modelpath = os.path.join(progpath, 'models')
resultpath = os.path.join(progpath, 'results')



###########################################################################################################################  organizing data
assert imdb.creatDataset(datasetpath, datasetname=datasetname) == True, "Failing to creat train/val/test sets"
if datasetname in ['cubbirds', 'vggaircraft']:
    data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((600,600)),
            transforms.RandomCrop((448, 448)),
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((600,600)),
            transforms.CenterCrop((448, 448)),
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
else:
    data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((image_size,image_size)),
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


####################################################################################################### constructing or loading model

    
model = sef.resnet50(pretrained=False, model_dir=modelzoopath, nparts=nparts, num_classes=num_classes, attention=attention_flag, device=device)
state_dict_path = os.path.join(modelzoopath, "resnet50-19c8e357.pth")

state_params = torch.load(state_dict_path)

# pop redundant params from laoded states
state_params.pop('fc.weight')
state_params.pop('fc.bias')

# modify output layer
in_channels = model.fc.in_features
new_fc = nn.Linear(in_channels, num_classes, bias=True)
model.fc = new_fc

# initializing model using pretrained params except the modified layers
model.load_state_dict(state_params, strict=False)
 

# tensorboard writer
images, _ = next(iter(dataloader['test']))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid)
writer.add_graph(model, images)

# to gpu if available
model.cuda(device)


########################################################################################################### creating loss functions
# cross entropy loss
cls_loss = nn.CrossEntropyLoss()

# semantic group loss
lmgm_loss = sef.LocalMaxGlobalMin(rho=lmgm, nchannels=512*4, nparts=nparts, device=device)

criterion = [cls_loss, lmgm_loss]



#################################################################################################################### creating optimizer
# optimizer
optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9)

# optimization scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)



######################################################################################################################### train model
isckpt = False  # set True to load learned models from checkpoint, defatult False

indfile = "{}: opt={}, lr={}, lmgm={}, nparts={}, entropy={}, soft={}, epochs={}, imgsz={}, batch_sz={}".format(
    datasetname, optmeth, lr, lmgm, nparts, entropy, soft, epochs, image_size, batchsize)
print("\n{}\n".format(indfile))
print("\n{}\n".format(indfile), file=logfile)


model, train_rsltparams = modellearning.train(
    model, dataloader, criterion, optimizer, scheduler, 
    datasetname=datasetname, isckpt=isckpt, epochs=epochs, 
    networkname=networkname, writer=writer, device=device, maxent_flag=maxent_flag,
    soft_weights=soft, entropy_weights=entropy, logfile=logfile)


train_rsltparams['imgsz'] = image_size
train_rsltparams['epochs'] = epochs
train_rsltparams['init_lr'] = lr
train_rsltparams['batch_sz'] = batchsize

print('\nBest epoch: {}'.format(train_rsltparams['best_epoch']))
print('\nBest epoch: {}'.format(train_rsltparams['best_epoch']), file=logfile)
print("\n{}\n".format(indfile))
print("\n{}\n".format(indfile), file=logfile)
print('\nWorking on cluster: {}\n'.format(device_name))

logfile.close()


#################################################################################################################### save model
torch.save({'model_params':model.state_dict(), 'train_params':train_rsltparams}, os.path.join(modelpath, modelname))

