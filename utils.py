import os
import numpy as np
import random
import torch.nn.functional as F
import torch
from sklearn import metrics
from datetime import datetime
from sklearn.metrics import f1_score,roc_auc_score
import matplotlib.pyplot as plt
import time

def setup_seed(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
def compute_metrics(test,pred):
    confusion_matrix=metrics.confusion_matrix(test,pred)
    tp=confusion_matrix[1][1]
    fp=confusion_matrix[0][1]
    fn=confusion_matrix[1][0]
    tn=confusion_matrix[0][0]
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    if tn == 0 and fp == 0:
        specificity = 0
    else:
        specificity = tn / (fp + tn)

    if (tp == 0 and fn == 0):
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    f1=f1_score(test,pred)
    auc=roc_auc_score(test,pred)
    return sensitivity,specificity,f1,auc

def plot_loss_acc_curve(val_losses,train_losses,val_acc,train_acc,dataname,fold_id):
    plt.figure(figsize=(15,5))
    # plt.figure(figsize=(6,8))

    plt.subplot(1, 2, 1)
    # plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses,label="val")
    plt.plot(train_losses,label="train")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    # plt.figure(figsize=(10,5))
    plt.title("Training and Validation Acc")
    plt.plot(val_acc,label="val")
    plt.plot(train_acc,label="train")
    plt.xlabel("epochs")
    plt.ylabel("Acc")
    plt.legend()

    plt.show()

    if_exist('Fig/{}/'.format(dataname,fold_id))
    plt.savefig('Fig/{}/fold_{}_loss.png'.format(dataname,fold_id))


def train_model(num_training,num_test,args, model, optimizer, train_data, test_data, fold_id, i, scheduler=None):
    max_acc=0.0
    val_losses = []
    train_losses = []
    train_acc=[]
    val_acc=[]
    patience=0
    min_loss = 1e10
    for epoch in range(args.epochs):
        loss_train,acc_train,sen_train,spe_train=train_epoch(train_data, model, optimizer,scheduler,args.b)
        train_losses.append(loss_train)
        train_acc.append(acc_train)

        test_loss,test_acc,test_sen,test_spe,test_f1,test_auc,Nattention,Tattention=compute_test(test_data,model)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
            'acc_train: {:.6f}'.format(acc_train),'sen_train: {:.6f}'.format(sen_train),
            'spe_train: {:.6f}'.format(spe_train),'loss_test: {:.6f}'.format(test_loss))
        val_losses.append(test_loss)
        val_acc.append(test_acc)
        
        if test_loss < min_loss:
            torch.save(model.state_dict(), 'ckpt/{}/{}_{}_fold_best_model.pth'.format(args.data, i, fold_id))
            print("Model saved at epoch{}".format(epoch))
            min_loss = test_loss
            patience = 0        
        else:
            patience += 1
        if patience == args.patience:
            break
    # plot_loss_acc_curve(val_losses,train_losses,val_acc,train_acc,args.target_dir,fold_id)


class BinaryFocalLoss(torch.nn.Module):
    """
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-5 # set '1e-4' when train with FP16
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss


def train_model_BinaryFocalLoss(num_training,num_test,args, model, optimizer, train_data, test_data, fold_id, i, scheduler=None):
    max_acc=0.0
    val_losses = []
    train_losses = []
    train_acc=[]
    val_acc=[]
    patience=0
    min_loss = 1e10
    for epoch in range(args.epochs):
        loss_train,acc_train,sen_train,spe_train=train_epoch_BinaryFocalLoss(train_data, model, optimizer,scheduler,args.b)
        train_losses.append(loss_train)
        train_acc.append(acc_train)
        
        #max acc
        test_loss,test_acc,test_sen,test_spe,test_f1,test_auc,Nattention,Tattention=compute_test_BinaryFocalLoss(test_data,model)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
            'acc_train: {:.6f}'.format(acc_train),'sen_train: {:.6f}'.format(sen_train),
            'spe_train: {:.6f}'.format(spe_train),'loss_test: {:.6f}'.format(test_loss))
        val_losses.append(test_loss)
        val_acc.append(test_acc)
        
        if test_loss < min_loss:
            torch.save(model.state_dict(), 'ckpt/{}/{}_{}_fold_best_model.pth'.format(args.data, i, fold_id))
            min_loss = test_loss
            patience = 0        
        else:
            patience += 1
        if patience == args.patience:
            break


def train_epoch_BinaryFocalLoss(dataloader, model, optimizer,scheduler=None,b=0):
    size = len(dataloader.dataset)
    y_list = []
    pred_list = []
    model.train()
    correct = 0
    loss_train = 0.0
    for batch,data in enumerate(dataloader):
        # Compute prediction error
        output,node_attention,time_attention = model(data)
        pred = output.max(dim=1)[1]
        loss = BinaryFocalLoss()(output, data[0].long())
        # flood= (loss-b).abs()+b
        # Backpropagation
        optimizer.zero_grad()
        # flood.backward()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
           scheduler.step()
        loss_train += loss.item()
        correct += pred.eq(data[0]).sum().item()
        pred_num = pred.detach().cpu().numpy()
        y_num = data[0].long().detach().cpu().numpy()
        for num in range(len(pred)):
            pred_list.append(pred_num[num])
            y_list.append(y_num[num])
    
    loss_train=loss_train/(batch+1)
    acc_train = correct / size
    sensitivity,specificity,_,_=compute_metrics(y_list,pred_list)    
    return loss_train,acc_train,sensitivity,specificity

def compute_test_BinaryFocalLoss(test_data, model):
    model.eval()
    size = len(test_data.dataset)
    y_list = []
    pred_list = []
    correct = 0.0
    loss_test = 0.0
    for i,data in enumerate(test_data):
        with torch.no_grad():
        # data = data.to(args.device)
            out,node_attention,time_attention = model(data)
            pred = out.max(dim=1)[1]
            # print(pred)
            correct += pred.eq(data[0]).sum().item()
            loss_test += BinaryFocalLoss()(out, data[0].long()).item()
            pred_num = pred.detach().cpu().numpy()
            y_num = data[0].detach().cpu().numpy()
            for num in range(len(pred)):
                pred_list.append(pred_num[num])
                y_list.append(y_num[num])
    loss_test=loss_test/(i+1)
    acc_test = correct / size
    sensitivity,specificity,f1,auc=compute_metrics(y_list,pred_list) 
    node_attention=node_attention.mean(dim=0)
    time_attention=time_attention.mean(dim=0)
    return loss_test,acc_test, sensitivity,specificity,f1,auc,node_attention.detach().cpu().numpy(),time_attention.detach().cpu().numpy()



def train_model_CrossEntropy(num_training,num_test,args, model, optimizer, train_data, test_data, fold_id,scheduler=None):
    max_acc=0.0
    val_losses = []
    train_losses = []
    train_acc=[]
    val_acc=[]
    patience=0
    min_loss = 1e10
    for epoch in range(args.epochs):
        loss_train,acc_train,sen_train,spe_train=train_epoch_CrossEntropy(train_data, model, optimizer,scheduler,args.b)
        train_losses.append(loss_train)
        train_acc.append(acc_train)
        
        #max acc
        test_loss,test_acc,test_sen,test_spe,test_f1,test_auc,Nattention,Tattention=compute_test_CrossEntropy(test_data,model)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
            'acc_train: {:.6f}'.format(acc_train),'sen_train: {:.6f}'.format(sen_train),
            'spe_train: {:.6f}'.format(spe_train),'loss_test: {:.6f}'.format(test_loss))
        val_losses.append(test_loss)
        val_acc.append(test_acc)
        
        if test_loss < min_loss:
            torch.save(model.state_dict(), 'ckpt/{}/{}_fold_best_model.pth'.format(args.target_dir,fold_id))
            min_loss = test_loss
            patience = 0        
        else:
            patience += 1
        if patience == args.patience:
            break
    plot_loss_acc_curve(val_losses,train_losses,val_acc,train_acc,args.target_dir,fold_id)

    # plot_loss_curve(val_losses,train_losses,args.target_dir,fold_id)
    # plot_acc_curve(val_acc,train_acc,args.target_dir,fold_id)

def train_epoch_CrossEntropy(dataloader, model, optimizer,scheduler=None,b=0):
    size = len(dataloader.dataset)
    y_list = []
    pred_list = []
    model.train()
    correct = 0
    loss_train = 0.0
    for batch,data in enumerate(dataloader):
        # Compute prediction error
        output,node_attention,time_attention = model(data)
        pred = output.max(dim=1)[1]
        # weights=[1.0/206.0,1.0/304.0]
        weights=[314.0/206.0,314.0/314.0]
        class_weights =torch.FloatTensor(weights).cuda()
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)(output, data[0].long())
        flood= (loss-b).abs()+b
        # Backpropagation
        # optimizer.zero_grad()
        flood.backward()
        # loss.backward()
        optimizer.step()
        if scheduler is not None:
           scheduler.step()
        loss_train += loss.item()
        correct += pred.eq(data[0]).sum().item()
        pred_num = pred.detach().cpu().numpy()
        y_num = data[0].long().detach().cpu().numpy()
        for num in range(len(pred)):
            pred_list.append(pred_num[num])
            y_list.append(y_num[num])
    #####
    loss_train=loss_train/(batch+1)
    acc_train = correct / size
    sensitivity,specificity,_,_=compute_metrics(y_list,pred_list)    
    return loss_train,acc_train,sensitivity,specificity

def compute_test_CrossEntropy(test_data, model):
    model.eval()
    size = len(test_data.dataset)
    y_list = []
    pred_list = []
    correct = 0.0
    loss_test = 0.0
    for i,data in enumerate(test_data):
        with torch.no_grad():
        # data = data.to(args.device)
            out,node_attention,time_attention = model(data)
            pred = out.max(dim=1)[1]
            # print(pred)
            correct += pred.eq(data[0]).sum().item()
            # weights=[1.0/206.0,1.0/304.0]
            weights=[314.0/206.0,314.0/314.0]
            class_weights =torch.FloatTensor(weights).cuda()
            loss_test += torch.nn.CrossEntropyLoss(weight=class_weights)(out, data[0].long()).item()
            pred_num = pred.detach().cpu().numpy()
            y_num = data[0].detach().cpu().numpy()
            for num in range(len(pred)):
                pred_list.append(pred_num[num])
                y_list.append(y_num[num])
    loss_test=loss_test/(i+1)
    acc_test = correct / size
    sensitivity,specificity,f1,auc=compute_metrics(y_list,pred_list) 
    node_attention=node_attention.mean(dim=0)
    time_attention=time_attention.mean(dim=0)
    return loss_test,acc_test, sensitivity,specificity,f1,auc,node_attention.detach().cpu().numpy(),time_attention.detach().cpu().numpy()



def train_epoch(dataloader, model, optimizer,scheduler=None,b=0):
    size = len(dataloader.dataset)
    y_list = []
    pred_list = []
    model.train()
    correct = 0
    loss_train = 0.0
    for batch,data in enumerate(dataloader):
        # Compute prediction error
        output,node_attention,time_attention = model(data)
        pred = output.max(dim=1)[1]
        loss = F.nll_loss(output, data[0].long())
        flood= (loss-b).abs()+b
        # Backpropagation
        optimizer.zero_grad()
        flood.backward()
        # loss.backward()
        optimizer.step()
        if scheduler is not None:
           scheduler.step()
        loss_train += loss.item()
        correct += pred.eq(data[0]).sum().item()
        pred_num = pred.detach().cpu().numpy()
        y_num = data[0].long().detach().cpu().numpy()
        for num in range(len(pred)):
            pred_list.append(pred_num[num])
            y_list.append(y_num[num])
    #####
    loss_train=loss_train/(batch+1)
    acc_train = correct / size
    sensitivity,specificity,_,_=compute_metrics(y_list,pred_list)    
    return loss_train,acc_train,sensitivity,specificity

def compute_test(test_data, model):
    model.eval()
    size = len(test_data.dataset)
    y_list = []
    pred_list = []
    correct = 0.0
    loss_test = 0.0
    for i,data in enumerate(test_data):
        with torch.no_grad():
        # data = data.to(args.device)
            out,node_attention,time_attention = model(data)
            pred = out.max(dim=1)[1]
            # print(pred)
            correct += pred.eq(data[0]).sum().item()
            loss_test += F.nll_loss(out, data[0].long()).item()
            pred_num = pred.detach().cpu().numpy()
            y_num = data[0].detach().cpu().numpy()
            for num in range(len(pred)):
                pred_list.append(pred_num[num])
                y_list.append(y_num[num])
    # 计算推理时间
    loss_test=loss_test/(i+1)
    acc_test = correct / size
    sensitivity,specificity,f1,auc=compute_metrics(y_list,pred_list) 
    node_attention=node_attention.mean(dim=0)
    time_attention=time_attention.mean(dim=0)
    return loss_test,acc_test, sensitivity,specificity,f1,auc,node_attention.detach().cpu().numpy(),time_attention.detach().cpu().numpy()

def save_std(args,acc_iter,sen_iter,spe_iter,f1_iter,auc_iter):
    with open('output/%s.txt' % args.data, 'a+') as f:
        f.write('%s  acc\t %.2f (± %.2f) sensitivity\t %.2f (± %.2f) specificity\t %.2f (± %.2f) f1\t %.2f (± %.2f) auc\t %.2f (± %.2f)\n' % (
                    str(datetime.now()),np.mean(acc_iter),np.std(acc_iter), np.mean(sen_iter), np.std(sen_iter),np.mean(spe_iter),np.std(spe_iter),np.mean(f1_iter),np.std(f1_iter),np.mean(auc_iter),np.std(auc_iter)))

def save_each_fold(args,test_acc,test_sen,test_spe,test_f1,test_auc,fold_id):
    with open('output/%s.txt' % args.data, 'a+') as f:
        f.write(' %s fold %dacc\t%.6f sensitivity\t%.6f specificity\t%.6f f1\t%.6f auc\t%.6f\n' % (str(datetime.now()),
                    fold_id,test_acc, test_sen, test_spe,test_f1,test_auc))

def if_exist(path):
    if os.path.isdir(path) == False:
        os.makedirs(path)

def save_para(file,total_params,args,num_training,num_test):
    file.write('total parameters are %d '%total_params)
    file.write('num_training is %d and num_test is %d \n'%(num_training,num_test))

    argsDict = args.__dict__
    for eachArg, value in argsDict.items():
        file.writelines(eachArg + ' : ' + str(value) + ' ')
    file.write('\n')
