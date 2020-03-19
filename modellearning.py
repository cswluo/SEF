import copy
import torch
import time
import torch.nn.functional as F
import pdb
from utils import modelserial
import torch.nn as nn
import os
import torch.utils.tensorboard as tb

from utils.misc import SoftCrossEntropy

# import multiprocessing
# torch.multiprocessing.set_start_method('spawn', True)
# torch.multiprocessing.freeze_support()


device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
# device = torch.device('cpu')
softmax = nn.Softmax(dim=-1)
logsoftmax = nn.LogSoftmax(dim=-1)
kldiv = nn.KLDivLoss(reduction='batchmean')


def train(model, dataloader, criterion, optimizer, scheduler, datasetname=None, isckpt=False, epochs=50, networkname=None, writer=None):

    output_log_file = open('./results/'+datasetname+'-'+networkname+'-'+time.strftime("%d-%b-%Y-%H:%M")+'.txt', 'w')
    
    if device.type == 'cuda':
        attention_flag = model.module.attention
    else:
        attention_flag = model.attention

    # get the size of train and evaluation data
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

            for inputs, labels in dataloader[phase]:
                #pdb.set_trace()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'trainval'):

                    if attention_flag:
                        outputs, attention_weights, _ = model(inputs)
                        logits = torch.sum(outputs, dim=0)
                        # prob = logits.clone()

                        
                        # compute prediction for each branch
                        prob_global = softmax(torch.squeeze(outputs[0]))
                        prob_part1 = softmax(torch.squeeze(outputs[1]))
                        prob_part2 = softmax(torch.squeeze(outputs[2]))
                        prob = torch.stack([prob_global, prob_part1, prob_part2], dim=0).max(dim=0)[0]

                        cls_loss_global = criterion[0](torch.squeeze(outputs[0]), labels)
                        cls_loss_part1 = criterion[0](torch.squeeze(outputs[1]), labels)
                        cls_loss_part2 = criterion[0](torch.squeeze(outputs[2]), labels)
                        cls_loss = cls_loss_global + cls_loss_part1 + cls_loss_part2
                        

                        log_prob_part1 = logsoftmax(torch.squeeze(outputs[1]))
                        log_prob_part2 = logsoftmax(torch.squeeze(outputs[2]))
                        klloss_part1 = kldiv(log_prob_part1, prob_global)
                        klloss_part2 = kldiv(log_prob_part2, prob_global)
                        klloss = klloss_part1 + klloss_part2


                        # reg loss
                        sparse_reg_loss = criterion[1](attention_weights)
                        # similarity_reg_loss = criterion[2](attention_weights)
                        reg_loss = sparse_reg_loss + klloss

                        

                    else:
                        outputs = model(inputs)
                        logits = outputs.clone()
                        prob = softmax(outputs)
                        sparse_reg_loss = torch.tensor(0.0, device=device)
                        # similarity_reg_loss = torch.tensor(0.0, device=device)
                        reg_loss = torch.tensor(0.0, device=device)
                        cls_loss = criterion[0](logits, labels)
                        
                    
                    _, preds = torch.max(prob, 1)   # the indeices of the largeset value in each row   

                    all_loss = cls_loss + reg_loss
                    
                    if phase == 'trainval':                       
                        all_loss.backward()
                        optimizer.step()

                # statistics
                running_cls_loss += (cls_loss.item()) * inputs.size(0)
                running_reg_loss += (reg_loss.item()) * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # log variables
                global_step += 1
                if global_step % 100 == 1 and writer is not None and phase is 'trainval':
                    batch_loss = cls_loss.item() + reg_loss.item()
                    writer.add_scalar('running loss/running_train_loss', batch_loss, global_step)
                    # writer.add_scalar('running loss/running_sim_reg_loss', similarity_reg_loss, global_step)
                    writer.add_scalar('running loss/running_sps_reg_loss', sparse_reg_loss, global_step)
                    writer.add_scalar('running loss/running_kl_part1_loss', klloss_part1, global_step)
                    writer.add_scalar('running loss/running_kl_part2_loss', klloss_part2, global_step)
                    writer.add_scalar('running loss/running_kl_reg_loss', klloss, global_step)
                    writer.add_scalar('running loss/running_cl_global_reg_loss', cls_loss_global, global_step)
                    writer.add_scalar('running loss/running_cl_part1_reg_loss', cls_loss_part1, global_step)
                    writer.add_scalar('running loss/running_cl_part2_reg_loss', cls_loss_part2, global_step)
                    # pdb.set_trace()
                    for name, param in model.named_parameters():
                        writer.add_histogram('params_in_running/'+name, param.data.clone().cpu().numpy(), global_step)     # global_step



            ############################################### for each epoch
            
            # epoch loss and accuracy
            epoch_loss = (running_cls_loss + running_reg_loss) / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # log variables for each epoch
            if writer is not None:
                if phase is 'trainval':
                    writer.add_scalar('epoch loss/train_epoch_loss', epoch_loss, epoch)        # global_step
                    writer.add_scalar('accuracy/train_epoch_acc', epoch_acc, epoch)          # global_step
                    for name, param in model.named_parameters():
                        writer.add_histogram('params_in_epoch/'+name, param.data.clone().cpu().numpy(), epoch)     # global_step
                elif phase is 'test':
                    writer.add_scalar('epoch loss/eval_epoch_loss', epoch_loss, epoch)         # global_step_resume
                    writer.add_scalar('accuracy/eval_epoch_acc', epoch_acc, epoch)          # global_step_resume

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), file=output_log_file)
            if phase == 'test': print('\n', file=output_log_file)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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
                                            'best_acc': best_acc}, datasetname+'-'+networkname)
        
        
        # the end of training on all epochs
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60), file=output_log_file)
    print('Best test Acc: {:4f}'.format(best_acc) , file=output_log_file)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best test Acc: {:4f}'.format(best_acc))

    rsltparams = dict()
    rsltparams['val_acc'] = best_acc.item()
    rsltparams['sparsity'] = criterion[1].rho
    # rsltparams['similarity'] = criterion[2].rho
    rsltparams['lr'] = optimizer.param_groups[0]['lr']
    rsltparams['best_epoch'] = best_epoch
    rsltparams['best_step'] = best_step

    # close output log file
    output_log_file.close()

    # load best model weights
    model.load_state_dict(best_model_params)
    return model, rsltparams






def eval(model, dataloader=None):

    if device.type == 'cuda':
        attention_flag = model.module.attention
    else:
        attention_flag = model.attention

    model.eval()
    datasize = len(dataloader.dataset)
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            if attention_flag:
                outputs, _, _ = model(inputs)
            else:
                outputs = model(inputs)
                    
            # _, preds = torch.max(outputs_ulti+outputs_plty+outputs_cmbn, 1)
            prob = softmax(outputs)
            # prob_nn = softmax(outputs_nn)
            # prob_pool = softmax(outputs_pool)
            _, preds = torch.max(prob, 1)
            # # outputs_ulti, outputs_plty, outputs_cmbn, _, _, _ = model(inputs)
            # if model.module.nparts > 1:
            #     outputs_ulti, _, _ = model(inputs)
            # else:
            #     outputs_ulti = model(inputs)
            # # preds = torch.argmax(outputs_ulti + outputs_plty + outputs_cmbn, dim=1)
            # preds = torch.argmax(outputs_ulti, dim=1)
        running_corrects += torch.sum(preds == labels.data)
    
    acc = torch.div(running_corrects.double(), datasize).item()
    print("Test Accuracy: {}".format(acc))

    rsltparams = dict()
    rsltparams['test_acc'] = acc
    
    return rsltparams


if __name__=='__main__':
    pass