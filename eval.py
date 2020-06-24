import os, sys, time
import pickle as pk
import numpy as np
from pprint import pprint
from utils import mydataloader
from utils import myimagefolder


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
progpath = os.path.dirname(os.path.realpath(__file__))      
sys.path.append(progpath)
import modellearning
import sef



eps = torch.finfo().eps
device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
device_name = device.type+':'+str(device.index) if device.type=='cuda' else 'cpu'

###########################################################################################################################
#
#  model zoo and dataset path
#
###########################################################################################################################
datapath = '/path/to/your/datasets'
modelzoopath = '/path/to/your/resnet/models'
sys.path.append(os.path.realpath(modelzoopath))
modelpath = os.path.join(progpath, 'models')
resultpath = os.path.join(progpath, 'results')




###########################################################################################################################
#
#  constructing loading models
#
###########################################################################################################################
modelname = r"your_learned_model"
load_params = torch.load(os.path.join(modelpath, modelname), map_location='cpu')
networkname = modelname.split('-')[1]


################################## loading from models trained on single gpu to initialize params
model_state_dict, train_params = load_params['model_params'], load_params['train_params']
pprint(train_params)

datasetname = modelname.split('-',1)[0]
nparts = train_params['nparts']
lmgm = train_params['lmgm']
entropy = train_params['entropy_weights']
soft = train_params['soft_weights']
batchsize = train_params['batch_sz']
imgsz = train_params['imgsz']
lr = train_params['init_lr']
if datasetname == 'cubbirds': num_classes = 200
if datasetname == 'vggaircraft': num_classes = 100
if datasetname == 'stdogs': num_classes = 120
if datasetname == 'stcars': num_classes = 196
attention_flag = True if nparts > 1 else False
netframe = 'resnet50' if networkname.find('50') > -1 else 'resnet18'

# resnet with attention    
model = sef.__dict__[netframe](pretrained=False, model_dir=modelzoopath, nparts=nparts, num_classes=num_classes, attention=attention_flag)

# initializing model using pretrained params except the modified layers
model.load_state_dict(model_state_dict, strict=True)
    


###########################################################################################################################
#
#  generating pytorch dataset and dataloader
#
###########################################################################################################################
datasetpath = os.path.join(datapath, datasetname)
assert imdb.creatDataset(datasetpath, datasetname=datasetname) == True, "Failing to creat train/val/test sets"
if datasetname in ['cubbirds', 'nabirds', 'vggaircraft']:
    data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((600,600)),
            transforms.RandomCrop((448, 448)),
            transforms.Resize((imgsz,imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((600,600)),
            transforms.CenterCrop((448, 448)),
            transforms.Resize((imgsz,imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
else:
    data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((imgsz,imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((imgsz,imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
test_transform = data_transform['test']




###########################################################################################################################
#
#  codes for model prediction and collecting correctly and wrongly predicted image names and labels.
#
###########################################################################################################################
testsplit = myimagefolder.ImageFolder(os.path.join(datasetpath, 'test'), data_transform['test'])
testloader = mydataloader.DataLoader(testsplit, batch_size=64, shuffle=False, num_workers=8)

testsplit_size = len(testsplit)
class_names = testsplit.classes
class_index = testsplit.class_to_idx
image_index = testsplit.imgs


log_items = r'{}-net{}-att{}-lmgm{}-entropy{}-soft{}-lr{}-imgsz{}-bsz{}.pkl'.format(
    datasetname, int(networkname[3:5]), nparts, lmgm, entropy, soft, lr, imgsz, batchsize)

if not os.path.exists(os.path.join(resultpath, log_items)):
    model.cuda(device)
    test_rsltparams = modellearning.eval(model, testloader, datasetname=datasetname, device=device)
    with open(os.path.join(resultpath, log_items), 'wb') as f:
        pk.dump({'acc': test_rsltparams['acc'], 'good_data': test_rsltparams['good_data'], 'bad_data':test_rsltparams['bad_data'], 'avg_acc':test_rsltparams['avg_acc']}, f)
else:
    with open(os.path.join(resultpath, log_items), 'rb') as f:
        test_rsltparams = pk.load(f)


print('General Acc: {}, Class Avg Acc: {}'.format(test_rsltparams['acc'], test_rsltparams['avg_acc']))


