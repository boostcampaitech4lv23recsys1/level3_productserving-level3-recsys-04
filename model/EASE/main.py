import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os

from model import EASE



''' global variables '''
############# Warning... 실행 전 반드시 설정 확인해주세요!! #############
path = '../../backend/app/models/data/'  # Data 저장 경로
data_type = 'rand'  # ['time', 'rand']
k = 20  # Should be synced with "k" in ease inference.py
thres = 2000  # Should be synced with "thres" in ease inference.py
output_dir = './output/'
output_csv_name = '20230202'  # output csv 이름 설정

lambda_ = 500  # EASE 모델 lambda 값 설정 (이미 만들어진 pickle 파일이 없을 경우에만 유효)
#######################################################################
try:  # data_type & 이미 저장된 pickle 파일 존재하는지 체크 (bool)
    is_pickle_exist = True if data_type=='time' and os.listdir( path + 'ease/' ) else False
except FileNotFoundError:
    is_pickle_exist = False
''''''


''' load data '''
train = pd.read_csv(path + f'train_{data_type}.csv')
test = pd.read_csv(path + f'test_{data_type}.csv')
''''''

''' model train '''
model = EASE(k, thres)

if is_pickle_exist:
    model.load_X_B_matrix(path)
else:
    model.fit(train, lambda_)
    if data_type == 'time':
        model.save_X_B_matrix(path)
''''''

''' predict '''
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
        with open(f'{path}/ease/ease-pred-{i}.pkl', 'rb') as f:
            pred_cur = pickle.load(f)
    else:
        X_cur = model.X[ start : end ]
        pred_cur = X_cur.dot(model.B)
        pred_cur = np.float16(pred_cur)  ## 용량 줄이기
        if data_type == 'time':
            with open(f'{path}/ease/ease-pred-{i}.pkl', 'wb') as f:
                pickle.dump(pred_cur, f, pickle.HIGHEST_PROTOCOL)
    
    pred_cur = model.predict(start, train_gbr[start:end], items_tot, pred_cur)
    predict = pd.concat([predict, pred_cur])
''''''

''' recall@k '''
predict = predict.reset_index(drop=True)
predict = predict.drop('score', axis = 1)
predict = predict.astype('int')

predict_user = predict.groupby('user_code')['rest_code'].apply(list)
answer_user = test.groupby('user_code')['rest_code'].apply(list)

predict_user = predict_user.reset_index(drop=True)
answer_user = answer_user.reset_index(drop=True)

# output csv 생성
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
predict_user.to_csv(f'{output_dir}ease_{data_type}_top{k}_{output_csv_name}.csv')
#################

_recall = []

for i, ans in enumerate(answer_user):
    a = 0
    for j in ans:
        if j in predict_user[i]:
            a += 1 
    _recall.append(a/2)

recall = sum(_recall) / len(_recall)
print(recall)
''''''
