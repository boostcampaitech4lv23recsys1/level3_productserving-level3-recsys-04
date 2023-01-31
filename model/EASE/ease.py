import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle


''' EASE model class '''
class EASE:
    def __init__(self):
        return
    

    def fit(self, df, lambda_: float = 500, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = df['user_code'], df['rest_code']
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['rating'].to_numpy() / df['rating'].max()
        )

        X = csr_matrix(
            (values, (users, items)),
            shape=( df['user_code'].max()+1, df['rest_code'].max()+1 ),
            dtype=np.float16
        )
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        # self.pred = X.dot(B)


    def predict(self, df, items, k):
        g = df.groupby('user_code')
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items, k) for user, group in g],
            )
        df = pd.concat(user_preds)
        return df


    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group['rest_code'])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "user_code": [user] * len(res),
                "rest_code": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values('score', ascending=False)
        return r
        

''' load data '''
path = '../../backend/app/models/data/'

train = pd.read_csv(path + 'train_time.csv')
test = pd.read_csv(path + 'test_time.csv')

'''
model train
'''
model = EASE()
model.fit(train)

with open('../data/ease-X-f16.pickle', 'rb') as f:
    X = pickle.load(f)
with open('../data/ease-B-f16.pickle', 'rb') as f:
    B = pickle.load(f)

'''
save pred matrix seperated
'''
thres = 2000
user_max = X.shape[0] - 1
for i in tqdm(range(user_max//thres + 1)):
    # with open(f'{path}ease/ease-pred-{i}.pickle', 'wb') as f:
    with open(f'../data/ease/ease-pred-{i}.pickle', 'wb') as f:
        XX = X[ i*thres : (i+1)*thres ]
        pred_cur = XX.dot(B)
        pred_cur = np.float16(pred_cur)
        pickle.dump(pred_cur, f, pickle.HIGHEST_PROTOCOL)


np.save('../data/ease-pred.npy', model.pred)


predict = model.predict(train,train['user'].unique(),train['item'].unique(),3)
predict = predict.drop('score',axis = 1)

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
recall
