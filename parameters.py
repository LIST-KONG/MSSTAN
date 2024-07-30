import argparse

parser = argparse.ArgumentParser()
####change dataname and window_size
parser.add_argument('--max_lr', type=float, default=0.003)
parser.add_argument('--b', type=float, default=0.5)
parser.add_argument('--data', type=str, default='ASD', help='path')
parser.add_argument('--target_dir', type=str, default='ASD', help='path')
parser.add_argument('--window_size', type=int, default=100,help='window_size.')
parser.add_argument('--step', type=int, default=2,help='step.')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

parser.add_argument('--total_window_size', type=int, default=230,help='window_size.')
parser.add_argument('--threshold', type=float, default=0.2,help='threshold.')
parser.add_argument('--feature', type=str, default='FC', choices=['BOLD', 'Degree', 'BoldCatDegree', 'FC'])
parser.add_argument('--nhead', type=int, default=1, help='n_head')
parser.add_argument('--source_dir', type=str, default="/data/ABIDE/",help='Dataset to use.')
##########################tune parameters#############################
parser.add_argument('--spatial_dropout', type=float, default=0,help='Dropout rate.')
parser.add_argument('--temporal_dropout', type=float, default=0,help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=500,help='Number of epochs to train.')

parser.add_argument('--dropout', type=float, default=0,help='Dropout rate (1 - keep probability).')
parser.add_argument('--em_dropout', type=float, default=0,help='Dropout rate (1 - keep probability).')
parser.add_argument('--temporal_nhead', type=int, default=1, help='n_head')    
parser.add_argument('--iter_time', type=int, default=10, help='itertime')
parser.add_argument('--lr', type=float, default=0.0003, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
##########################rarely modified#############################
parser.add_argument('--node_num', type=int, default=90,help='node num.')
parser.add_argument('--seed', type=int, default=125, help='Random seed.')
parser.add_argument('--no_folds', type=int, default=10, help='K-fold cross-validation')
parser.add_argument('--nclass', type=int, default=2, help='nhid')
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--nhid', type=int, default=32, help='nhid')
parser.add_argument('--link-pred', action='store_true', default=False, help='Enable Link Prediction Loss')

