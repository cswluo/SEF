import os, sys, time
import pickle as pk
import pdb

import torch, torchvision
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
import torch.multiprocessing as mlp
import torch.utils.tensorboard as tb


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
batchsize = 32
nthreads = 8
attention_flag = True

networkname = 'resnet18attention'
cluster = False
mymodels = myresnetmodels if networkname.find('resnet')>-1 else myvggnetmodels
writer = tb.SummaryWriter(log_dir='./runs/'+datasetname+'/'+networkname+time.strftime("%d-%b-%Y"))


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

################################### constructing or loading model

############ number of attentions for different datasets
if datasetname in ['stcars', 'vggaircraft', 'nabirds', 'cubbirds']:
    nparts = 2
elif datasetname in ['stdogs']:
    nparts = 3
else:
    nparts = 1  # number of parts you want to use for your dataset


############ constructing models with different backbones
if networkname.find('vgg') > -1:


    model = mymodels.vgg19_bn(pretrained=False, modelpath=modelzoopath, nparts=nparts, num_classes=num_classes,sharemask=True)
    state_dict_path = os.path.join(modelzoopath, "vgg19_bn-c79401a0.pth")
    pretrained_model = os.path.split(state_dict_path)[-1]
    state_params = torch.load(state_dict_path, map_location=device)

    model.load_state_dict(state_params, strict=False)
    model = init_normal_branch.initVggnet(pretrained_model, state_params, model)

elif networkname.find('resnet') > -1:

    # two methods for initializing models from pretrained models
    # 1. setting pretrained=True and then modifying your own models
    # 2. setting pretrained=False and then modifying your own models before initializing it using loaded parameter states
    # here we choose the 2nd method.

    # resnet with attention
    model = mymodels.resnet18(pretrained=False, model_dir=modelzoopath, nparts=nparts, num_classes=num_classes, attention=attention_flag)

    # pure resnet
    # model = mymodels.resnet18(pretrained=False, model_dir=modelzoopath, nparts=0, num_classes=num_classes, writer=writer)

    state_dict_path = os.path.join(modelzoopath, "resnet18-5c106cde.pth")
    state_params = torch.load(state_dict_path, map_location=device)

    # pop redundant params from laoded states
    state_params.pop('fc.weight')
    state_params.pop('fc.bias')
    # print(state_params.keys())


    # modify output layer
    in_channels = model.fc.in_features
    new_fc = nn.Linear(in_channels, num_classes, bias=True)
    model.fc = new_fc

    
    # initializing model using pretrained params except the modified layers
    model.load_state_dict(state_params, strict=False)

 
else:
    pass

# tensorboard writer
images, _ = next(iter(dataloader['test']))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid)
writer.add_graph(model, images)


################ move model into GPUs
if torch.cuda.device_count() > 0:
    model = nn.DataParallel(model)
model.to(device)

# pdb.set_trace()
############################################# creating loss functions

# cross entropy loss
cls_loss = nn.CrossEntropyLoss()

# sparsity induced regularization loss
sparsity = 1e-3
if networkname.find('vgg') > -1:
    sparse_loss = myvggmodels.SparseLoss(rho=sparsity, nparts=nparts)
elif networkname.find('resnet') > -1:
    sparse_loss = myresnetmodels.SparseLoss(rho=sparsity, nparts=nparts)
else:
    pass 

# similarity induced regularization loss
# distance = 0.0
# if networkname.find('vgg') > -1:
#     dist_loss = myvggmodels.DistLoss(rho=distance, nparts=nparts)
# elif networkname.find('resnet') > -1:
#     dist_loss = myresnetmodels.DistLoss(rho=distance, nparts=nparts)
# else:
#     pass 


criterion = [cls_loss, sparse_loss]


############################################ creating optimizer
lr = 1e-2
plr = 1e-3
optmeth = 'sgd'
optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer = opt.SGD([{'params': model.module.features.parameters()},
#                      {'params': model.module.vgg_normal.parameters()}, 
#                      {'params': model.module.global_classifier.parameters(), 'lr': 0},
#                      {'params': model.module.mask_filters.parameters(), 'lr': plr},
#                      {'params': model.module.gen_classifier.parameters(), 'lr': plr},
#                      {'params': model.module.gen_filters.parameters(), 'lr': plr},
#                      {'params': model.module.part_nn.parameters(), 'lr': 0}], 
#                      lr=lr, momentum=0.9)


# creating optimization scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# training the model
epochs = 30
isckpt = False  # set True to load learned models from checkpoint, defatult False

# print parameters
print("{}: lr: {}, plr: {}, sparsity: {}, nparts: {}, epochs: {}".format(optmeth, lr, plr, sparsity, nparts, epochs))

model, train_rsltparams = modellearning.train(
    model, dataloader, criterion, optimizer, scheduler, datasetname=datasetname, isckpt=isckpt, epochs=epochs, 
    networkname=networkname, writer=writer)

# writer.export_scalars_to_json("./runs/"+datasetname+".json")
# writer.close()

#### save model
timeflag = time.strftime("%d-%b-%Y-%H:%M")
modelpath = './models'
modelname = r"{}-{}-parts{}-sps{}-{}-lr{}-{}-{}.model".format(
    datasetname, networkname, nparts, sparsity, optmeth, lr, image_size, timeflag)
torch.save(model.state_dict(), os.path.join(modelpath, modelname))


########################### evaluation
#testsplit = datasets.ImageFolder(os.path.join(datasetpath, 'test'), data_transform['val'])
#testloader = torch.utils.data.DataLoader(testsplit, batch_size=64, shuffle=False, num_workers=8)
#test_rsltparams = modellearning.eval(model, testloader)


########################### record results
#filename = r"parts{}-sc{}_{}_{}-{}{}-wobnrelu-SENet50-448-new.pkl".format(nparts, gamma1, gamma2, gamma3, optmeth, lr)
#rsltpath = os.path.join(progpath, 'results', filename)
#with open(rsltpath, 'wb') as f:
#    pk.dump({'train': train_rsltparams, 'test': test_rsltparams}, f)
