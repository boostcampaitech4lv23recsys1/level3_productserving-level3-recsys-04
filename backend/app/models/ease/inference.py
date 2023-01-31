import numpy as np
import pandas as pd

import pickle
import argparse


class EASE:
    def __init__(self, pred):
        self.pred = pred
        breakpoint()
        self.item_size = pred.shape[0]


    def predict_for_user(self, user_seq, candidate, k):
        watched = set(user_seq)
        candidates = [i for i in range(1, self.item_size) if i not in watched and i in candidate]
        pred = self.pred
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "rest_code": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values('score', ascending=False)
        return list(r['rest_code'])


def recommend(user, user_seq, candidate, k):
    ''' parser '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/", type=str)  # app/models/data/
    parser.add_argument("--thres", default="2000", type=int)  # sync with ease model train
    args = parser.parse_args()
    ''''''

    ''' load pred matrix '''
    ith_file, user_idx = divmod( user, args.thres )
    with open(args.data_dir + f'ease/ease-pred-{ith_file}', 'rb') as f:
        pred = pickle.load(f)
    pred = pred[user_idx]
    ''''''

    ''' recommend top k '''
    model = EASE(pred)
    top_k = model.predict_for_user(user_seq, candidate, k)
    return top_k
    ''''''
