from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy import io
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
import pickle
from glob import glob

def set_kfold(args,dataset,fold_id,total_data,random_seed):
    inst = KFold(n_splits=args.no_folds, shuffle=True, random_state=random_seed)#args.seed
    KFolds = list(inst.split(np.arange(total_data)))
    training_idx, test_idx=KFolds[fold_id]
    num_training=training_idx.size
    num_test=test_idx.size
    training_set=Subset(dataset,training_idx)
    test_set=Subset(dataset,test_idx)
    train_data = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(test_set, batch_size=min(num_test,args.batch_size), shuffle=False)#num_test
    return num_training,num_test,train_data,test_data

def decide_dataset(path):
    if "ABIDE" in path:
        label_list=['HC','ASD']
        label_set = {'HC': 0, 'ASD': 1}
    return label_list,label_set

def read_dataset_fc_regionSeries(args,path, label_files, files):
    subj_fc_dir = os.path.join(path, label_files, files, 'calStaticRes.pkl')
    with open(subj_fc_dir, "rb") as f:
        subj_fc_mat = pickle.load(f).source_mat[:args.total_window_size,:90]
    print("reading data " + subj_fc_dir)
    return subj_fc_mat

#利用阈值计算邻接矩阵
def get_Pearson_fc(sub_region_series,threshold):
    subj_fc_adj = np.corrcoef(np.transpose(sub_region_series))
    # subj_fc_adj = subj_fc_adj - np.diag(np.diag(subj_fc_adj))
    subj_fc_adj_up=subj_fc_adj[np.triu_indices(90,k=1)]
    subj_fc_adj_list = subj_fc_adj_up.reshape((-1))
    thindex = int(threshold * subj_fc_adj_list.shape[0])
    thremax = subj_fc_adj_list[subj_fc_adj_list.argsort()[-1 * thindex-1]]
    subj_fc_adj_t = np.zeros((90, 90))
    subj_fc_adj_t[subj_fc_adj > thremax] = 1
    subj_fc_adj=subj_fc_adj_t
    return subj_fc_adj

def get_fc_degree(subj_fc_adj):
    rowsum = np.array(subj_fc_adj.sum(1))
    N = np.diag(rowsum) 
    return N   

def max_min_norm(sub_region_series):
    subj_fc_mat_list = sub_region_series.reshape((-1))
    subj_fc_feature = (sub_region_series - min(subj_fc_mat_list)) / (max(subj_fc_mat_list) - min(subj_fc_mat_list))
    return subj_fc_feature


class MyDynamicDataSet(Dataset):
    def __init__(self,args):     

        label = []     
        each_sub_adj=[]
        each_sub_feature=[]

        threshold = args.threshold

        dataset_dir=args.source_dir

        label_list,label_set=decide_dataset(dataset_dir)

        for label_files in label_list:
            list = os.listdir(os.path.join(dataset_dir, label_files))
            for files in list:
                fc_adj = []
                fc_features = []

                sliding_window_size=args.window_size
                subj_fc_mat = read_dataset_fc_regionSeries(args,dataset_dir, label_files, files)
                total_window_size=args.total_window_size

                for j in range(0, total_window_size - sliding_window_size, args.step):
                    sub_region_series = subj_fc_mat[j:j + sliding_window_size, :]

                    subj_fc_adj=get_Pearson_fc(sub_region_series,threshold)
                    fc_adj.append(subj_fc_adj)

                    feature_selection=args.feature
                    if feature_selection=='BOLD':
                        subj_fc_feature=max_min_norm(sub_region_series)
                        fc_features.append(np.transpose(subj_fc_feature)) 
                    elif feature_selection=='Degree':
                        fc_degree=get_fc_degree(subj_fc_adj)
                        fc_features.append(fc_degree) 
                    elif feature_selection=='BoldCatDegree':
                        subj_fc_feature=max_min_norm(sub_region_series)
                        fc_degree=get_fc_degree(subj_fc_adj)
                        bold_C_degree=np.concatenate((subj_fc_feature,fc_degree),0)
                        fc_features.append(np.transpose(bold_C_degree)) 
                    elif feature_selection=='FC':
                        fc_features.append(np.corrcoef(np.transpose(sub_region_series)))
                each_sub_adj.append(np.array(fc_adj))
                each_sub_feature.append(np.array(fc_features))
                label.append(label_set[label_files])
                
        
        self.label = np.array(label)
        self.fc_adj = np.array(each_sub_adj)
        self.fc_features = np.array(each_sub_feature)
        self.length = self.fc_features.shape[0]
        print('The size of this dataset is %d'%self.length)

    def __getitem__(self, mask):
        label = self.label[mask]
        fc_adj = self.fc_adj[mask]
        fc_features=self.fc_features[mask]
        return label, fc_adj,fc_features

    def __len__(self):
        return self.length


