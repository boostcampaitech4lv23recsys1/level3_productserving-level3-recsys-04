import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from scipy import sparse
import torch.nn.functional as F
import bottleneck as bn


class DataLoader():
    def __init__(self, path):            
        self.pro_dir = os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files do not exist. Run data.py"
        self.n_items = self.load_n_items()        
    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        elif datatype == 'sub':
            return self._load_train_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test, submission]")  
    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sub_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items        
    def _load_train_data(self, datatype = 'train'):
        path = os.path.join(self.pro_dir, f'{datatype}.csv')            
        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1
        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                (rows, cols)), dtype='float64',
                                shape=(n_users, self.n_items))
        return data
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))
        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']
        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te
    
class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        h = self.decode(z)
        return h, mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h
    def init_weights(self):
        for layer in self.q_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)        
        for layer in self.p_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
           
def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())    
def predict(self, model, data_tr, is_VAE=False):
    model.eval()
    global update_count
    items = []
    with torch.no_grad():
        for start_idx in range(data_tr.shape[0]):
            data = data_tr[start_idx]
            data_tensor = self.naive_sparse2tensor(data).to(device)  
            if is_VAE :        
                recon_batch, mu, logvar = model(data_tensor)
            else:
                recon_batch = model(data_tensor)
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf                
            for rec in recon_batch:
                up = np.argpartition(rec, -10)[-10:].tolist()
                items.extend(up)
    return items       
def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    return BCE      
def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD       


##TODO 아래코드 어케 처리함?
    model = torch.load('MultiDAE.pt') 
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
    parser.add_argument('--data', type=str, default='',
                        help='Movielens dataset location')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args([])
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        args.cuda = True
    device = torch.device("cuda" if args.cuda else "cpu")
    loader = DataLoader(args.data)
    sub_data = loader.load_data('sub')
    id2item = pd.read_csv("id2item.csv")
    id2user = pd.read_csv("id2user.csv")
    user2id = id2user.set_index('2').to_dict()['1']
    id2item = id2item.set_index('1').to_dict()['2']
    user2rec_list = predict(model, sub_data[user2id["5c944530817bf510af65740e"]], is_VAE=False) #TODO 여기에 유저 이름 넣어주기만 하면 됨
    user2rec_list = [id2item[x] for x in user2rec_list]
    return user2rec_list