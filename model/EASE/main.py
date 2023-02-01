import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from os import listdir

from model import EASE
        

''' global variables '''
path = '../../backend/app/models/data/'
is_pickle_exist = True if listdir( path + 'ease/' ) else False
k = 20  # Should be synced with "k" in ease inference.py
thres = 2000  # Should be synced with "thres" in ease inference.py
''''''

''' load data '''
train = pd.read_csv(path + 'train_time.csv')
test = pd.read_csv(path + 'test_time.csv')
''''''

''' model train '''
model = EASE(thres)

if is_pickle_exist:
    model.load_X_B_matrix(path)
else:
    model.fit(train)
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
        with open(f'{path}/ease/ease-pred-{i}.pickle', 'rb') as f:
            pred_cur = pickle.load(f)
    else:
        with open(f'{path}/ease/ease-pred-{i}.pickle', 'wb') as f:
            X_cur = model.X[ start : end ]
            pred_cur = X_cur.dot(model.B)
            pred_cur = np.float16(pred_cur)
            pickle.dump(pred_cur, f, pickle.HIGHEST_PROTOCOL)
    
    pred_cur = model.predict(start, train_gbr[start:end], train['rest_code'].unique(), pred_cur, k)
    predict = pd.concat([predict, pred_cur])
''''''

''' recall@k '''
predcit = predict.reset_index(drop=True)
predict = predict.drop('score', axis = 1)

predict_user = predict.groupby('user_code')['rest_code'].apply(list)
answer_user = test.groupby('user_code')['rest_code'].apply(list)

predict_user = predict_user.reset_index(drop=True)
answer_user = answer_user.reset_index(drop=True)

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
