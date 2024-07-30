import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from parameters import parser
import torch
import torch.optim as optim
import numpy as np
from model import *
from utils import *
import scipy   
from torch import load,save   
from load_data import *


args = parser.parse_args()
if_exist('output/')
if_exist('datasets/')
target_dir=args.target_dir
file=open('output/%s.txt' % args.data, 'a+')

def main():
    setup_seed(args)
    if os.path.isfile(f'datasets/{args.data}.pth'):
        dataset = load(f'datasets/{args.data}.pth')    
    else:
        dataset = MyDynamicDataSet(args)
        save(dataset,os.path.join(f'datasets/{args.data}.pth'),pickle_protocol=4)
    graph_num=dataset.fc_adj.shape[1]

    if args.cuda:
        dataset.fc_features = torch.FloatTensor(dataset.fc_features).cuda()
        dataset.fc_adj = torch.FloatTensor(dataset.fc_adj).cuda()
        dataset.label=torch.FloatTensor(dataset.label).cuda()

    total_data=dataset.length 
    print('total dataset size is {}'.format(total_data))

    acc_set,sen_set,spe_set,f1_set,auc_set=[[] for i in range(5)]

    for i in range(args.iter_time):
        random_seed=[25, 50, 100, 125, 150, 175, 200, 225, 250, 275]
        acc,sensitivity,specificity,f1,auc,node_attr_set,time_attr_set=[[] for _ in range(7)]

        for fold_id in range(args.no_folds):
            num_training,num_test,train_data,test_data = set_kfold(args,dataset,fold_id,total_data,random_seed[i])
            model=MSSTAN(args,graph_num)

            model.cuda()

            print ("model = ...")
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f'{total_params:,} total parameters.')

            optimizer=optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=total_data, pct_start=0.2, div_factor=args.max_lr/args.lr, final_div_factor=1000)
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=int(num_training/args.batch_size), pct_start=0.2, div_factor=args.max_lr/args.lr, final_div_factor=1000)

            # train_model(num_training,num_test,args, model, optimizer, train_data, test_data, fold_id, i, scheduler=None)
            # train_model_CrossEntropy(num_training,num_test,args, model, optimizer, train_data, test_data, fold_id,scheduler=None)
            train_model_BinaryFocalLoss(num_training,num_test,args, model, optimizer, train_data, test_data, fold_id, i, scheduler=None)

            model.load_state_dict(torch.load('ckpt/{}/{}_{}_fold_best_model.pth'.format(args.data, i, fold_id)))
            
            # _,test_acc,test_sen,test_spe,test_f1,test_auc,Nattention,Tattention=compute_test(test_data,model)
            _,test_acc,test_sen,test_spe,test_f1,test_auc,Nattention,Tattention=compute_test_BinaryFocalLoss(test_data,model)

            save_each_fold(args,test_acc,test_sen,test_spe,test_f1,test_auc,fold_id) 
            acc.append(test_acc*100)
            sensitivity.append(test_sen*100)
            specificity.append(test_spe*100)
            f1.append(test_f1*100)
            auc.append(test_auc*100)
            node_attr_set.append(Nattention)
            time_attr_set.append(Tattention)


        acc_set.append(np.mean(acc))
        sen_set.append(np.mean(sensitivity))
        spe_set.append(np.mean(specificity))
        f1_set.append(np.mean(f1))
        auc_set.append(np.mean(auc))
        save_std(args,acc,sensitivity,specificity,f1,auc)

    save_std(args,acc_set,sen_set,spe_set,f1_set,auc_set) 
    save_para(file,total_params,args,num_training,num_test)

main()
file.close()
