import copy
import torch
import time
import torch.nn.functional as F
import pdb
from utils import modelserial
import torch.nn as nn
import os
import torch.utils.tensorboard as tb


softmax = nn.Softmax(dim=-1)
logsoftmax = nn.LogSoftmax(dim=-1)
kldiv = nn.KLDivLoss(reduction='batchmean')


def train(model, dataloader, criterion, optimizer, scheduler, datasetname=None, isckpt=False, epochs=50, networkname=None, writer=None, maxent_flag=False, device='cpu', **penalty):

    output_log_file = penalty['logfile']
    nparts = model.nparts
    attention_flag = model.attention
    
    if isinstance(dataloader, dict):
        dataset_sizes = {x: len(dataloader[x].dataset) for x in dataloader.keys()}
        print(dataset_sizes)
    else:
        dataset_size = len(dataloader.dataset)

    if not isinstance(criterion, list):
        criterion = [criterion]

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    global_step = 0
    global_step_resume = 0
    best_epoch = 0
    best_step = 0
    start_epoch = -1
    

    if isckpt:
        checkpoint = modelserial.loadCheckpoint(datasetname+'-'+networkname)

        # records for the stopping epoch
        start_epoch = checkpoint['epoch']
        global_step_resume = checkpoint['global_step']
        model.load_state_dict(checkpoint['state_dict'])

        # records for the epoch with the best performance
        best_model_params = checkpoint['best_state_dict']
        best_acc = checkpoint['best_acc']
        best_epoch = checkpoint['best_epoch']
        optimizer.param_groups[0]['lr'] = checkpoint['current_lr']

    since = time.time()
    for epoch in range(start_epoch+1, epochs):

        # print to file
        print('Epoch {}/{}'.format(epoch, epochs), file=output_log_file)
        print('-' * 10, file=output_log_file)

        # print to terminal
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)


        for phase in ['trainval', 'test']:
            if phase == 'trainval':
                # scheduler.step()
                model.train()  # Set model to training mode
                global_step = global_step_resume
            else:
                model.eval()   # Set model to evaluate mode
                global_step_resume = global_step

            running_cls_loss = 0.0
            running_reg_loss = 0.0
            running_corrects = 0.0
            running_corrects_parts = [0.0] * nparts
            epoch_acc_parts = [0.0] * nparts


            for inputs, labels in dataloader[phase]:
                inputs = inputs.cuda(device)
                labels = labels.cuda(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'trainval'):

                    if attention_flag:
                        # outputs are logits from linear models
                        xglobal, xlocal, xcosin, _ = model(inputs)
                        probs = softmax(xglobal)                    
                        cls_loss = criterion[0](xglobal, labels)

                        ############################################################## prediction

                        # prediction of every  branch
                        probl, predl, logprobl = [], [], []
                        for i in range(nparts):
                            probl.append(softmax(torch.squeeze(xlocal[i])))
                            predl.append(torch.max(probl[i], 1)[-1])
                            logprobl.append(logsoftmax(torch.squeeze(xlocal[i])))


                        ############################################################### regularization

                        logprobs = logsoftmax(xglobal)
                        entropy_loss = penalty['entropy_weights'] * torch.mul(probs, logprobs).sum().div(inputs.size(0))
                        soft_loss_list = []
                        for i in range(nparts):
                            soft_loss_list.append(torch.mul(torch.neg(probs), logprobl[i]).sum().div(inputs.size(0)))
                        soft_loss = penalty['soft_weights'] * sum(soft_loss_list).div(nparts)

                        # regularization loss
                        lmgm_reg_loss = criterion[1](xcosin)
                        reg_loss = lmgm_reg_loss + entropy_loss + soft_loss


                    else:
                        outputs = model(inputs)
                        probs = softmax(outputs)
                        cls_loss = criterion[0](outputs, labels)
                        if maxent_flag:
                            logprobs = logsoftmax(outputs)
                            reg_loss = torch.mul(probs, logprobs).sum().neg().div(inputs.size(0))
                        else:
                            reg_loss = torch.tensor(0.0)

 
                    _, preds = torch.max(probs, 1)   # the indeices of the largeset value in each row   

                    all_loss = cls_loss + reg_loss
                    
                    if phase == 'trainval':                       
                        all_loss.backward()
                        optimizer.step()

                # statistics
                running_cls_loss += (cls_loss.item()) * inputs.size(0)
                running_reg_loss += (reg_loss.item()) * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if attention_flag:
                    for i in range(nparts):
                        running_corrects_parts[i] += torch.sum(predl[i] == labels.data)
                    
                # log variables
                global_step += 1
                if global_step % 100 == 1 and writer is not None and phase is 'trainval':
                    batch_loss = cls_loss.item() + reg_loss.item() 
                    writer.add_scalar('running loss/running_train_loss', batch_loss, global_step)
                    writer.add_scalar('running loss/running_cls_loss', cls_loss, global_step) 
                    if attention_flag:                     
                        writer.add_scalar('running loss/running_lmgm_reg_loss', lmgm_reg_loss, global_step)  
                        writer.add_scalar('running loss/running_entropy_reg_loss', entropy_loss, global_step)  
                        writer.add_scalar('running loss/running_soft_reg_loss', soft_loss, global_step)  
                    elif maxent_flag:
                        writer.add_scalar('running loss/running_maxent_reg_loss', reg_loss, global_step)  
                    for name, param in model.named_parameters():
                        writer.add_histogram('params_in_running/'+name, param.data.clone().cpu().numpy(), global_step)     # global_step



            ############################################### for each epoch
            
            # epoch loss and accuracy
            epoch_loss = running_cls_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if attention_flag:
                for i in range(nparts):
                    epoch_acc_parts[i] = running_corrects_parts[i].double() / dataset_sizes[phase]


            # log variables for each epoch
            if writer is not None:
                if phase is 'trainval':
                    writer.add_scalar('epoch loss/train_epoch_loss', epoch_loss, epoch)        # global_step
                    writer.add_scalar('accuracy/train_epoch_acc', epoch_acc, epoch)          # global_step
                    if attention_flag:
                        for i in range(nparts):
                            writer.add_scalar('accuracy/train_acc_part{}_acc'.format(i), epoch_acc_parts[i], epoch) 
                    for name, param in model.named_parameters():
                        writer.add_histogram('params_in_epoch/'+name, param.data.clone().cpu().numpy(), epoch)     # global_step
                elif phase is 'test':
                    writer.add_scalar('epoch loss/eval_epoch_loss', epoch_loss, epoch)         # global_step_resume
                    writer.add_scalar('accuracy/eval_epoch_acc', epoch_acc, epoch)          # global_step_resume
                    if attention_flag:
                        for i in range(nparts):
                            writer.add_scalar('accuracy/eval_acc_part{}_acc'.format(i), epoch_acc_parts[i], epoch) 

            # print to log file
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), file=output_log_file)
            if phase == 'trainval': print('current lr: {}'.format(optimizer.param_groups[0]['lr']), file=output_log_file)
            if phase == 'test': print('\n', file=output_log_file)

            # print to terminal
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'trainval': print('current lr: {}'.format(optimizer.param_groups[0]['lr']))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_step = global_step_resume
                best_model_params = copy.deepcopy(model.state_dict())

            if phase == 'test' and epoch % 5 == 1:
                modelserial.saveCheckpoint({'epoch': epoch,
                                            'global_step': global_step,
                                            'state_dict': model.state_dict(),
                                            'best_epoch': best_epoch,
                                            'best_state_dict': best_model_params,
                                            'best_acc': best_acc, 
                                            'current_lr': optimizer.param_groups[0]['lr']},datasetname+'-'+networkname)
        
        # adjust learning rate after each epoch
        scheduler.step()

        
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60), file=output_log_file)
    print('Best test Acc: {:4f}'.format(best_acc) , file=output_log_file)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best test Acc: {:4f}'.format(best_acc))


    # recording training params
    rsltparams = dict()
    rsltparams['datasetname'] = datasetname
    rsltparams['nparts'] = model.nparts
    rsltparams['val_acc'] = best_acc.item()
    rsltparams['lmgm'] = criterion[1].rho
    rsltparams['lr'] = optimizer.param_groups[0]['lr']
    rsltparams['best_epoch'] = best_epoch
    rsltparams['best_step'] = best_step
    rsltparams['soft_weights'] = penalty['soft_weights']
    rsltparams['entropy_weights'] = penalty['entropy_weights']

    # load best model weights
    model.load_state_dict(best_model_params)
    return model, rsltparams


def eval(model, dataloader=None, device='cpu', datasetname=None):

    if not datasetname or datasetname not in ['cubbirds', 'stcars', 'stdogs', 'vggaircraft', 'nabirds']:
        print("illegal dataset")
        return

    attention_flag = model.attention
    model.eval()
    datasize = len(dataloader.dataset)
    running_corrects = 0
    good_data = []
    bad_data = []
    num_label_counts = dict()
    pred_label_counts = dict()

    for paths, inputs, labels in dataloader:

        if datasetname == 'vggaircraft':
            for label in labels.data:
                num_label_counts.setdefault(label.item(), 0)
                num_label_counts[label.item()] += 1

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if attention_flag:
                outputs, _, _, _ = model(inputs)
            else:
                outputs = model(inputs)

        probs = softmax(outputs)
        _, preds = torch.max(probs, 1) 

        if datasetname == 'vggaircraft':
            for i, label in enumerate(preds.data):
                if label == labels[i]:
                    pred_label_counts.setdefault(label.item(), 0)
                    pred_label_counts[label.item()] += 1

        running_corrects += torch.sum(preds == labels.data)

        # record paths and labels
        good_mask = preds == labels.data
        bad_mask = torch.logical_not(good_mask)
        good_index = good_mask.nonzero()
        bad_index = bad_mask.nonzero()
        for idx in good_index:
            good_data.append((paths[idx], labels[idx].item()))
        for idx in bad_index:
            bad_data.append((paths[idx], labels[idx].item()))

    
    acc = torch.div(running_corrects.double(), datasize).item()
    avg_acc = 0.0
    print("General Accuracy: {}".format(acc))

    if datasetname == 'vggaircraft':
        running_corrects = 0
        for key in pred_label_counts.keys():
            running_corrects += pred_label_counts[key] / num_label_counts[key]  
        avg_acc = running_corrects / len(num_label_counts)
        print("{}: Class Average Accuracy: {}".format(datasetname, avg_acc))

    rsltparams = dict()
    rsltparams['acc'] = acc
    rsltparams['avg_acc'] = avg_acc
    rsltparams['good_data'] = good_data
    rsltparams['bad_data'] = bad_data
    
    return rsltparams

if __name__=='__main__':
    pass
