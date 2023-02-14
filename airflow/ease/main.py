import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from datetime import date

from ease.model import EASE
from sasrec.utils import personalizeion

import mlflow


def ease_main():
    '''
    global variables
    '''
    ############# Warning... 실행 전 반드시 설정 확인해주세요!! #############
    csv_input_path = '/opt/ml/input/project/airflow/dags/data/'  # input csv 경로
    # csv_input_path = '/opt/ml/input/project/backend/app/models/data/'  # input csv 경로
    csv_output_path = '/opt/ml/input/project/airflow/dags/output/'  # output csv 저장 경로
    cur_date = str(date.today())  # "2023-02-04"
    pkl_path = '/opt/ml/input/project/airflow/dags/data/'  # pickle 파일 저장&로드 경로

    data_type = 'time'  # ['time', 'rand']
    k = 20  # Should be synced with "k" in ease inference.py
    thres = 2000  # Should be synced with "thres" in ease inference.py

    lambda_ = 500  # EASE 모델 lambda 값 설정 (이미 만들어진 pickle 파일이 없을 경우에만 유효)
    #######################################################################
    try:  # data_type & 이미 저장된 pickle 파일 존재하는지 체크 (bool)
        is_pickle_exist = True if data_type=='time' and os.listdir( pkl_path + 'ease/' ) else False
    except FileNotFoundError:
        os.mkdir(pkl_path + 'ease')
        is_pickle_exist = False

    '''
    load data
    '''
    train = pd.read_csv(csv_input_path + f'train_{data_type}.csv')
    test = pd.read_csv(csv_input_path + f'test_{data_type}.csv')

    with mlflow.start_run():
        '''
        model train
        '''
        model = EASE(k, thres)

        if is_pickle_exist:
            model.load_X_B_matrix(pkl_path)
        else:
            model.fit(train, lambda_)
            if data_type == 'time':
                model.save_X_B_matrix(pkl_path)

        '''
        predict
        '''
        user_max = model.X.shape[0] - 1
        items_tot = train['rest_code'].unique()
        train_gbr = train.groupby('user_code')['rest_code'].apply(set)
        predict = pd.DataFrame(
            {
                "user_code": [],
                "rest_code": [],
                "score": [],
            }
        )

        for i in tqdm(range( user_max//thres + 1 )):
            start = i*thres; end = (i+1)*thres
            end = end if end < user_max else user_max+1

            if is_pickle_exist:
                with open(pkl_path + f'ease/ease-pred-{i}.pkl', 'rb') as f:
                    pred_cur = pickle.load(f)
            else:
                X_cur = model.X[ start : end ]
                pred_cur = X_cur.dot(model.B)
                pred_cur = np.float16(pred_cur)  ## 용량 줄이기
                if data_type == 'time':
                    with open(pkl_path + f'ease/ease-pred-{i}.pkl', 'wb') as f:
                        pickle.dump(pred_cur, f, pickle.HIGHEST_PROTOCOL)
            
            pred_cur = model.predict(start, train_gbr[start:end], items_tot, pred_cur)
            predict = pd.concat([predict, pred_cur])

        '''
        preprocess answer & predict
        '''
        predict = predict.reset_index(drop=True)
        predict = predict.drop('score', axis = 1)
        predict = predict.astype('int')

        predict_user = predict.groupby('user_code')['rest_code'].apply(list)
        answer_user = test.groupby('user_code')['rest_code'].apply(list)

        predict_user = predict_user.reset_index()
        predict_user.columns = ['index', 'pred']
        answer_user = answer_user.reset_index(drop=True)

        # output csv 생성
        if not os.path.exists(csv_output_path):
            os.mkdir(csv_output_path)
        predict_user.to_csv(f'{csv_output_path}ease-{data_type}-{cur_date}.csv')
        

        '''
        recall@k
        '''
        _recall = []

        for i, ans in enumerate(answer_user):
            a = 0
            for j in ans:
                if j in predict_user['pred'][i]:
                    a += 1 
            _recall.append(a/2)

        recall = sum(_recall) / len(_recall)

        '''
        personalization score
        '''
        per_score = personalizeion(predict_user)

        '''
        mlflow logging
        '''
        mlflow.log_params({
            "ease_lambda": lambda_,
            "ease_thres": thres,
            "ease_k": k,
            "ease_data_type": data_type
        })
        mlflow.log_metrics({
            "ease_recall": recall,
            "ease_personalization": per_score
        })

    mlflow.end_run()

    return recall, per_score


if __name__ == '__main__':
    ease_main()
