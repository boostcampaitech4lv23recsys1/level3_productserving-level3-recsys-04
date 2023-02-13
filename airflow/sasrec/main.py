import argparse
import os
import pandas as pd
import numpy as np
from datetime import date

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from .datasets import SASRecTrainDataset
from .models import S3RecModel
from .utils import (
    check_path,
    get_user_seqs,
    set_seed,
    personalizeion,
)
from .trainers import (
    iteration,
    test_score
)

import mlflow

from box import Box


# 0.01687, 25에폭(time)
def sasrec_main():
    print("!1111")
    parser = argparse.ArgumentParser()
    args = {
        "data_dir" : "/opt/ml/input/project/airflow/dags/data/",
        "output_dir" : "/opt/ml/input/project/airflow/dags/output/",
        "data_name" : str(date.today()),
        "data_type" : "time",
        "model_name" : "SASRec",
        "model_save" : "/opt/ml/input/project/airflow/dags/data/",
        "hidden_size" : 128,
        "num_hidden_layers" : 2,
        "num_attention_heads" : 2,
        "hidden_act" : "gelu",
        "attention_probs_dropout_prob" : 0.2,
        "hidden_dropout_prob" : 0.3,
        "initializer_range" : 0.02,
        "max_seq_length" : 150,
        "lr" : 0.001,
        "batch_size" : 256,
        "epochs" : 1,
        "log_freq" : 1,
        "seed" : 42,
        "weight_decay" : 1e-6,
        "adam_beta1" : 0.7,
        "adam_beta2" : 0.9999,
        "gpu_id" : "0",        
    }
    args = Box(args)
    # 시드 고정 (utils.py 내 함수 존재)
    set_seed(args.seed)
    # output 폴더가 존재하는지 체크. 존재하지 않는다면 만들어줍니다. (utils.py 내 함수 존재)
    check_path(args.output_dir)
    print("!1112")
    
    # GPU 관련 설정 해줍니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 데이터 파일 불러오는 경로 설정합니다.
    train = pd.read_csv(args.data_dir + f'train_{args.data_type}.csv')    
    test = pd.read_csv(args.data_dir + f'test_{args.data_type}.csv')
   
    
    # 자세한건 get_user_seqs 함수(utils.py) 내에 써놨습니다.
    user_seq, test_lines, max_item, train_matrix = get_user_seqs(
        train, test
    )

    # 음식점 종류를 저장합니다.
    args.item_list = train['rest_code'].unique()

    # item id의 가장 큰 값 저장합니다.
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    
    # save model args, (model_name : Finetune_full, data_name : Ml, output_dir : output/)
    args_str = f"{args.model_name}-{args.data_type}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    # user * item 메트릭스.
    args.train_matrix = train_matrix 
    
    # SASRecDataset 클래스를 불러옵니다. (datasets.py 내 존재)
    # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    # output : user_id(유저번호), input_ids(item), target_pos(item), target_neg(item), answer(test_item)
    train_dataset = SASRecTrainDataset(args, user_seq)

    # RandomSampler : 데이터 셋을 랜덤하게 섞어줍니다. 인덱스를 반환해줍니다.
    train_sampler = RandomSampler(train_dataset)

    # 모델 학습을 하기 위한 데이터 로더를 만듭니다. 랜덤으로 섞고 배치 단위(defalut : 256)로 출력합니다.
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    test_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=args.batch_size * 2
    )
    print("!1113")
    '''
    mlflow run
    '''
    with mlflow.start_run():
        # S3RecModel 모델을 불러옵니다. (models.py 내 존재)
        model = S3RecModel(args=args)
        # 모델을 GPU에 실어요.
        model = model.to(args.device)

        for epoch in range(args.epochs):
            iteration(args, epoch, train_dataloader, model)
            if epoch % 5 == 0:
                scores, pred_list = test_score(args, epoch, test_dataloader, model, test_lines)
                mlflow.log_metric("sasrec_recall", scores)
                print("recall_k = ", scores)

        pred_df = pd.DataFrame(pred_list)
        pred_df['pred'] = pred_df.apply(lambda x : [x[i] for i in range(0,20)], axis = 1)
        pred_df = pred_df.reset_index()
        pred_df = pred_df[['index','pred']]
        per_score = personalizeion(pred_df)
        pred_df.to_csv(args.output_dir + f'{args.model_name}-{args.data_type}-{args.data_name}.csv', index = False)
        torch.save(model.state_dict(), os.path.join(args.model_save, f'{args.model_name}-{args.data_type}.pt'))

        # mlflow logging
        mlflow.log_params({
            "sasrec_data_type": args.data_type,
            "sasrec_max_seq_length": args.max_seq_length,            
            "sasrec_batch_size": args.batch_size,
            "sasrec_hidden_size": args.hidden_size,
            "sasrec_lr": args.lr,
            "sasrec_epochs": args.epochs
        })
        mlflow.log_metric("sasrec_personalization", per_score)
        mlflow.pytorch.log_model(model, "sasrec_model")
    
    mlflow.end_run()
    print("!1115")
    return scores, per_score


if __name__ == "__main__":
    sasrec_main()
