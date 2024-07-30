import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SelfAttentionPooling
from einops import rearrange
from torch.nn.parameter import Parameter
from einops import repeat
import math
from reformer_pytorch import Reformer,LSHSelfAttention


class MaskedAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        scores=scores+ mask.unsqueeze(dim=1)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        # temp=mask.unsqueeze(dim=1)
        # p_attn=p_attn * mask.unsqueeze(dim=1)
        #################

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MaskedMultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = MaskedAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x),attn

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MaskedTransformer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    #https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/transformer.py
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MaskedMultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        # self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.norm_1 = LayerNorm(hidden)
        self.norm_2 = LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)
        self._reset_parameters()
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        x,attn=self.attention.forward(x, x, x, mask=mask)
        x=x + self.dropout(self.norm_1(x))
        x=self.feed_forward(x)
        x=x + self.dropout(self.norm_2(x))
        return self.dropout(x),attn


class MSSTAN(nn.Module):
    def __init__(self, args,graph_num):
        super(MSSTAN, self).__init__()

        if args.feature=='BoldCatDegree':
            input_dim=args.node_num+args.window_size
        elif args.feature=='BOLD':
            input_dim=args.window_size
        else:
            input_dim=args.node_num    

        spatial_dim=input_dim
        temporal_dim=spatial_dim*2

        if graph_num%2==0:
            bucket_size=int(graph_num/2)
        else:
            bucket_size=int((graph_num+1)/2)

        self.spatial_transformer_1=MaskedTransformer(spatial_dim,args.nhead,2*spatial_dim,dropout=args.spatial_dropout)

        self.temporal_transformer=LSHSelfAttention(
                                dim = temporal_dim,
                                heads = args.nhead,
                                bucket_size = bucket_size,
                                n_hashes = 8,
                                causal = False
                            )
        self.temporal_transformer_2=LSHSelfAttention(
                                dim = temporal_dim,
                                heads = args.nhead,
                                bucket_size = bucket_size,
                                n_hashes = 8,
                                causal = False
                            )
        self.temporal_transformer_3=LSHSelfAttention(
                                dim = temporal_dim,
                                heads = args.nhead,
                                bucket_size = bucket_size,
                                n_hashes = 8,
                                causal = False
                            )
        
        self.pool1 = SelfAttentionPooling(spatial_dim, 0.3)
        self.pool2 = SelfAttentionPooling(spatial_dim, 0.5)
        self.pool3 = SelfAttentionPooling(spatial_dim, 0.7)

        self.fc1 = nn.Linear(temporal_dim, args.nhid)
        self.fc2 = nn.Linear(args.nhid, args.nhid // 2)
        self.fc3 = nn.Linear(args.nhid // 2, args.nclass)  
        self.dropout = args.dropout
        self.em_dropout = args.em_dropout
   
    def forward(self, data):
        padding=torch.zeros([len(data[0]),1,90,90]).cuda()
        data[1]=torch.cat([data[1],padding],dim=1)
        data[2]=torch.cat([data[2],padding],dim=1)

        x,adj,batch=data[2],data[1],len(data[0])
        graph_num=x.shape[1]

        x=rearrange(x, 'b t n c -> (b t) n c')
        adj=rearrange(adj,'b t n m -> (b t) n m' )

        x,sattention = self.spatial_transformer_1(x,mask=adj)

        pool1= self.pool1(adj,x)
        pool2= self.pool2(adj,x)
        pool3= self.pool3(adj,x)

        global_pool1 = torch.cat(
            [torch.mean(pool1,dim=1),
             torch.max(pool1,dim=1)[0]],
            dim=1)
        global_pool2 = torch.cat(
            [torch.mean(pool2,dim=1),
             torch.max(pool2,dim=1)[0]],
            dim=1)
        global_pool3 = torch.cat(
            [torch.mean(pool3,dim=1),
             torch.max(pool3,dim=1)[0]],
            dim=1)
        tattention = 0

        g1=self.temporal_transformer(global_pool1.view(batch,graph_num,-1))
        g2=self.temporal_transformer_2(global_pool2.view(batch,graph_num,-1))
        g3=self.temporal_transformer_3(global_pool3.view(batch,graph_num,-1))

        x=g1[0]+g2[0]+g3[0]

        x=x.mean(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x,torch.squeeze(sattention),torch.squeeze(torch.FloatTensor(tattention))


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class LayerGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(LayerGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x
