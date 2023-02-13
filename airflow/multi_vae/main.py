# !pip install python-box
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from box import Box

import os
import time
from datetime import date
from tqdm import tqdm

from multi_vae.model import (
    MakeMatrixDataSet,
    AEDataSet,
    MultiVAE,
    LossFunc
)
from sasrec.utils import personalizeion

import warnings
warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=True)

import mlflow


def get_recallk(pred_list, true_list):
    a = 0
    for pred in pred_list:
        if pred in true_list:
            a +=1
    recall = a / len(true_list)
    return recall


# 학습함수 - train
def train(update_count, config, model, criterion, optimizer, data_loader, make_matrix_data_set):
    model.train()
    loss_val = 0
    for users in data_loader:
        mat = make_matrix_data_set.make_matrix(users)
        mat = mat.to(config.device)

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
    return loss_val, update_count


# 학습함수 - evaluate
def evaluate(config, model, data_loader, user_valid, make_matrix_data_set):
    model.eval()
    RECALL_K = 0.0  # Recall@K

    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users)
            mat = mat.to(config.device)

            recon_mat = model(mat)
            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim = 1)

            for user, rec in zip(users, rec_list):
                uv = user_valid[user.item()]
                up = rec[-config.K:].cpu().numpy().tolist()
                RECALL_K += get_recallk( pred_list = up, true_list = uv )
    
    RECALL_K /= len(data_loader.dataset)
    return RECALL_K


def submission(config, model, data_loader, make_matrix_data_set):
    model.eval()
    submission = {}
    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users)
            mat = mat.to(config.device)

            recon_mat = model(mat)
            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim = 1)

            for user, rec in zip(users, rec_list):
                up = rec[-config.K:].cpu().numpy().tolist()
                submission[user] = up
    return submission


def multivae_main():
    '''
    환경변수 설정
    '''
    config = {
        'data_path' : '/opt/ml/input/project/backend/app/models/data/multivae' ,  # 데이터 경로
        'data_type' : 'time' ,  # rand or time
        'save' : '/opt/ml/input/project/backend/app/models/data/multivae',
        'output_path' : '/opt/ml/input/project/airflow/dags/output',

        'p_dims': [100, 400],
        'dropout_rate' : 0.5,
        'weight_decay' : 0.01,
        'valid_samples' : 2,
        'seed' : 22,
        'anneal_cap' : 0.2,
        'total_anneal_steps' : 200000,
        
        'lr' : 0.005,
        'batch_size' : 500,
        'num_epochs' : 1,
        'num_workers' : 2,
    }
    config = Box(config)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.K = 20
    # config.data_path = "/opt/ml/input/project/airflow/data/"

    '''
    데이터 로드 & 생성
    '''
    if not os.path.exists(config.data_path + f'MVtrain_{config.data_type}.csv'):
        train_df = pd.read_csv(config.data_path + f'train_{config.data_type}.csv')
        train_df = train_df.drop(columns='date')
        train_df.columns = ['user','item','user_idx','item_idx']
        train_df.to_csv(config.data_path + f'MVtrain_{config.data_type}.csv', index=False)

        valid_df = pd.read_csv(config.data_path + f'test_{config.data_type}.csv')
        valid_df.columns = ['user','item','user_idx','item_idx']
        valid_df.to_csv(config.data_path + f'MVtest_{config.data_type}.csv', index=False)

    print('file get success!')

    make_matrix_data_set = MakeMatrixDataSet(config = config)
    _, user_valid = make_matrix_data_set.get_train_valid_data()

    ae_dataset = AEDataSet( num_user = make_matrix_data_set.num_user )

    print('Dataset finished')

    data_loader = DataLoader(
        ae_dataset,
        batch_size = config.batch_size, 
        shuffle = True, 
        pin_memory = True,
        num_workers = config.num_workers,
    )

    with mlflow.start_run():
        model = MultiVAE(
            p_dims = config.p_dims + [make_matrix_data_set.num_item], 
            dropout_rate = config.dropout_rate
        ).to(config.device)

        criterion = LossFunc(loss_type = 'Multinomial', model_type = 'VAE')
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        print('model create finished!!!')

        best_recall = 0
        loss_list = []

        recall_list = []
        earlystop =0 

        print('train start!!')

        update_count = 1

        for epoch in tqdm(range(1, config.num_epochs + 1)):
            start = time.time()

            train_loss, update_count = train(
                update_count,
                config = config,
                model = model, 
                criterion = criterion, 
                optimizer = optimizer, 
                data_loader = data_loader,
                make_matrix_data_set = make_matrix_data_set,
            )
            recall = evaluate(
                config = config,
                model = model, 
                data_loader = data_loader,
                user_valid = user_valid,
                make_matrix_data_set = make_matrix_data_set,
            )

            loss_list.append(train_loss)
            recall_list.append(recall)
            end = time.time()
            time_taken = end - start

            print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f} |Recall@{config.K}: {recall:.5f}|time taken : {time_taken:.1f}')

            mlflow.log_metric("multivae_recall", recall)

            if recall > best_recall:
                torch.save( model.state_dict(), f'{config.save}-{config.data_type}.pt' )
                best_recall = recall
                earlystop = 0
            else: 
                earlystop +=1
            
            if earlystop == 10:
                break
        
        print('train done!!')

        sub = submission(
            config = config,
            model = model, 
            data_loader = data_loader,
            make_matrix_data_set = make_matrix_data_set
        )
        submission_file = pd.DataFrame.from_dict(sub, orient='index')
        submission_file.to_csv(config.output_path + f'multivae-{config.data_type}-{str(date.today())}.csv', index = False)  # multivae-time-2023-02-04.csv

        print('submission done!!')

        mlflow.log_params({
            "multivae_data_type": config.data_type,
            "multivae_top_K": config.K,
            "multivae_batch_size": config.batch_size,
            "multivae_p_dims": config.p_dims,
            "multivae_dropout_rate": config.dropout_rate,
            "multivae_data_type": config.data_type,
            "multivae_num_epochs": config.num_epochs
        })
        mlflow.pytorch.log_model(model, "sasrec_model")

        print('mlflow logging done!!')

    mlflow.end_run()

    return recall_list[-1], 0


if __name__ == "__main__":
    multivae_main()
