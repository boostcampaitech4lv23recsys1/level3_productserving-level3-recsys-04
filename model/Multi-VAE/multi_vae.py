# !pip install python-box
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from box import Box

import warnings

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=True)

config = {
    'data_path' : "/opt/ml/input/project/model/data" , # 데이터 경로

    'p_dims': [100, 400],
    'dropout_rate' : 0.5,
    'weight_decay' : 0.01,
    'valid_samples' : 2,
    'seed' : 22,
    'anneal_cap' : 0.2,
    'total_anneal_steps' : 200000,
    'save' : '/opt/ml/input/project/model/Multi-VAE/model.pt',
    
    'lr' : 0.005,
    'batch_size' : 500,
    'num_epochs' : 50,
    'num_workers' : 2,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Box(config)







# 데이터셋
class MakeMatrixDataSet():
    """
    MatrixDataSet 생성
    """
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(os.path.join(self.config.data_path, 'MVtrain_time.csv'))
        self.val = pd.read_csv(os.path.join(self.config.data_path, 'MVtest_time.csv'))
        # self.item_encoder, self.item_decoder = self.generate_encoder_decoder('item')
        # self.user_encoder, self.user_decoder = self.generate_encoder_decoder('user')
        self.num_item, self.num_user = sorted(self.df['item_idx'].unique())[-1]+1,sorted(self.df['user_idx'].unique())[-1]+1

        # self.df['item_idx'] = self.df['item'].apply(lambda x : self.item_encoder[x])
        # self.df['user_idx'] = self.df['user'].apply(lambda x : self.user_encoder[x])

        self.user_train, self.user_valid = self.generate_sequence_data()

    # def generate_encoder_decoder(self, col : str) -> dict:
    #     """
    #     encoder, decoder 생성

    #     Args:
    #         col (str): 생성할 columns 명
    #     Returns:
    #         dict: 생성된 user encoder, decoder
    #     """

    #     encoder = {}
    #     decoder = {}
    #     ids = self.df[col].unique()

    #     for idx, _id in enumerate(ids):
    #         encoder[_id] = idx
    #         decoder[idx] = _id

    #     return encoder, decoder
    
    def generate_sequence_data(self) -> dict:
        """
        sequence_data 생성

        Returns:
            dict: train user sequence / valid user sequence
        """
        users = defaultdict(list)
        v_users = defaultdict(list)
        user_train = {}
        user_valid = {}
        for user, item in zip(self.df['user_idx'], self.df['item_idx']):
            users[user].append(item)
        for user, item in zip(self.val['user_idx'], self.val['item_idx']):
            v_users[user].append(item)
        
        for user in users:
            np.random.seed(self.config.seed)

            user_total = users[user]
            val_total = v_users[user]
            # valid = np.random.choice(user_total, size = self.config.valid_samples, replace = False).tolist()
            valid = list(set(val_total))
            train = list(set(user_total))


            user_train[user] = train
            user_valid[user] = valid # valid_samples 개수 만큼 검증에 활용 (현재 Task와 가장 유사하게)

        return user_train, user_valid
    
    def get_train_valid_data(self):
        return self.user_train, self.user_valid

    def make_matrix(self, user_list, train = True):
        """
        user_item_dict를 바탕으로 행렬 생성
        """
        mat = torch.zeros(size = (user_list.size(0), self.num_item))
        for idx, user in enumerate(user_list):
            if train:
                mat[idx, self.user_train[user.item()]] = 1
            else:
                mat[idx, self.user_train[user.item()] + self.user_valid[user.item()]] = 1
        return mat
    

class AEDataSet(Dataset):
    def __init__(self, num_user):
        self.num_user = num_user
        self.users = [i for i in range(num_user)]

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx): 
        user = self.users[idx]
        return torch.LongTensor([user])
    

# 모델
class MultiVAE(nn.Module):

    def __init__(self, p_dims, dropout_rate = 0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]

        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])

        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = nn.Dropout(dropout_rate)
        self.init_weights()
    
    def forward(self, input, loss = False):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        h = self.decode(z)
        if loss:
            return h, mu, logvar
        else:
            return h
    
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
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiDAE(nn.Module):
    
    def __init__(self, p_dims, dropout_rate = 0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout_rate)
        
        self.init_weights()
    
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


# 학습함수
class LossFunc(nn.Module):

    def __init__(self, loss_type = 'Multinomial', model_type = None):
        super(LossFunc, self).__init__()
        self.loss_type = loss_type
        self.model_type = model_type

    def forward(self, recon_x = None, x = None, mu = None, logvar = None, anneal = None):
        if self.loss_type == 'Gaussian':
            loss = self.Gaussian(recon_x, x)
        elif self.loss_type == 'Logistic':
            loss = self.Logistic(recon_x, x)
        elif self.loss_type == 'Multinomial':
            loss = self.Multinomial(recon_x, x)
        
        if self.model_type == 'VAE':
            KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = loss + anneal * KLD
        
        return loss

    def Gaussian(self, recon_x, x):
        gaussian = F.mse_loss(recon_x, x)
        return gaussian

    def Logistic(self, recon_x, x):
        logistic = F.binary_cross_entropy(recon_x.sigmoid(), x, reduction='none').sum(1).mean()
        return logistic

    def Multinomial(self, recon_x, x):
        multinomial = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        return multinomial
    

def get_ndcg(pred_list, true_list):
    idcg = sum((1 / np.log2(rank + 2) for rank in range(1, len(pred_list))))
    dcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            dcg += 1 / np.log2(rank + 2)
    ndcg = dcg / idcg
    return ndcg

def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit

def get_recallk(pred_list, true_list, K):
    a = 0
    for pred in pred_list:
        if pred in true_list:
            a +=1
    recall = a / len(true_list)
    return recall

# 학습함수 - train

def train(model, criterion, optimizer, data_loader, make_matrix_data_set, config):
    global update_count
    model.train()
    loss_val = 0
    for users in data_loader:
        mat = make_matrix_data_set.make_matrix(users)
        mat = mat.to(device)

        if criterion.model_type == 'VAE':
            anneal = min(config.anneal_cap, 1. * update_count / config.total_anneal_steps)
            update_count += 1
            recon_mat, mu, logvar = model(mat, loss = True)
            
            optimizer.zero_grad()
            loss = criterion(recon_x = recon_mat, x = mat, mu = mu, logvar = logvar, anneal = anneal)

        else:
            recon_mat = model(mat)
            optimizer.zero_grad()
            loss = criterion(recon_x = recon_mat, x = mat)

        loss_val += loss.item()

        loss.backward()
        optimizer.step()

    loss_val /= len(data_loader)

    return loss_val

# 학습함수 - evaluate

def evaluate(model, data_loader, user_train, user_valid, make_matrix_data_set, K = 20):
    model.eval()

    # NDCG = 0.0 # NDCG@K
    # HIT = 0.0 # HIT@K
    RECALL_K =0.0 #Recall@10

    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users)
            mat = mat.to(device)

            recon_mat = model(mat)
            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim = 1)

            for user, rec in zip(users, rec_list):
                uv = user_valid[user.item()]
                up = rec[-K:].cpu().numpy().tolist()
                # NDCG += get_ndcg(pred_list = up, true_list = uv)
                # HIT += get_hit(pred_list = up, true_list = uv)
                RECALL_K +=get_recallk(pred_list = up, true_list = uv, K = K )
                
    # NDCG /= len(data_loader.dataset)
    # HIT /= len(data_loader.dataset)
    RECALL_K /= len(data_loader.dataset)

    return RECALL_K


if __name__ == "__main__":
    print("it's working don't worry ")
    # 데이터
    #multivae 용 csv 생성
    file_path = "/opt/ml/input/project/model/data/MVtrain_time.csv"

    if not os.path.exists(file_path):
        train_df = pd.read_csv('/opt/ml/input/project/model/data/train_time.csv')
        train_df = train_df.drop(columns='date')
        train_df.columns=['user','item','user_idx','item_idx']
        train_df.to_csv('/opt/ml/input/project/model/data/MVtrain_time.csv',index=False)

        valid_df = pd.read_csv('/opt/ml/input/project/model/data/test_time.csv')
        valid_df.columns=['user','item','user_idx','item_idx']
        valid_df.to_csv('/opt/ml/input/project/model/data/MVtest_time.csv',index=False)

    print('file get success!')

    # 학습
    make_matrix_data_set = MakeMatrixDataSet(config = config)
    user_train, user_valid = make_matrix_data_set.get_train_valid_data()

    ae_dataset = AEDataSet(
    num_user = make_matrix_data_set.num_user,
    )

    print('Dataset finished')
    data_loader = DataLoader(
    ae_dataset,
    batch_size = config.batch_size, 
    shuffle = True, 
    pin_memory = True,
    num_workers = config.num_workers,
    )

    loss_dict = {}
    # ndcg_dict = {}
    # hit_dict = {}
    recall_dict = {}

    model = MultiVAE(
        p_dims = config.p_dims + [make_matrix_data_set.num_item], 
        dropout_rate = config.dropout_rate).to(device)

    criterion = LossFunc(loss_type = 'Multinomial', model_type = 'VAE')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print('model create finished!!!')
    best_recall = 0
    update_count = 1
    loss_list = []
    # ndcg_list = []
    # hit_list = []
    recall_list = []
    K = 20

    print('train start!!')
    for epoch in range(1, config.num_epochs + 1):
        start = time.time()

        train_loss = train(
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            data_loader = data_loader,
            make_matrix_data_set = make_matrix_data_set,
            config = config,
            )
        
        recall= evaluate(
            model = model, 
            data_loader = data_loader,
            user_train = user_train,
            user_valid = user_valid,
            make_matrix_data_set = make_matrix_data_set,
            K = K
            )

        loss_list.append(train_loss)
        # ndcg_list.append(ndcg)
        # hit_list.append(hit)
        recall_list.append(recall)
        end = time.time()
        time_taken = end - start

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f} |Recall@{K}: {recall:.5f}|time taken : {time_taken:.1f}')


        if recall > best_recall:
            with open(config.save, 'wb') as f:
                torch.save(model, f)
            best_recall = recall

    print('!!!!done!!!!')