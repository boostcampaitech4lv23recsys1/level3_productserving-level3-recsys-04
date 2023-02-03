import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
import pickle


''' EASE model class '''
class EASE:
    def __init__(self, k, thres):
        self.k = k
        self.thres = thres
    

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
            shape=( df['user_code'].max()+1, df['rest_code'].max()+1 )
        )
        self.X = X.astype(np.float16)  # 용량 줄이기

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = np.float16(B)  # 용량 줄이기
        # self.pred = X.dot(B)  # 용량 부족으로 인해 다른 방식으로 처리


    def save_X_B_matrix(self, path):
        """_summary_
        self.fit 를 통해 계산된 X와 B 행렬을 저장합니다.
        Args:
            path (_type_): 저장 저장 경로 (str)
        """
        with open(path + 'ease-X.pkl', 'wb') as f:
            pickle.dump(self.X, f, pickle.HIGHEST_PROTOCOL)
        with open(path + 'ease-B.pkl', 'wb') as f:
            pickle.dump(self.B, f, pickle.HIGHEST_PROTOCOL)


    def load_X_B_matrix(self, path):
        """_summary_
        미리 저장해 둔 X와 B 행렬을 load 합니다.
        Args:
            path (_type_): 파일 로드 경로 (str)
        """
        with open(path + 'ease-X.pkl', 'rb') as f:
            self.X = pickle.load(f)
        with open(path + 'ease-B.pkl', 'rb') as f:
            self.B = pickle.load(f)

    
    def predict(self, start, watched, items, pred):
        """_summary_
        start ~ start+thred 구간의 user들에 대해, pred 행렬을 바탕으로 k개 추천
        Args:
            start (_type_): 해당 번호의 user부터 predict_for_user 함수에서 user 단위로 추천합니다. (int)
            watched (_type_): start ~ start+thres 사이 유저들의 음식점 방문 기록 (pd.DataFrame)
            items (_type_): 전체 음식점 code (numpy.ndarray)
            pred (_type_): start ~ start+thres 사이 유저들의 pred 행렬 조각
            k (_type_): 추천 개수 (int)

        Returns:
            _type_: _description_
        """
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, watch, pred[user % self.thres, :], items, self.k) for user, watch in enumerate(watched, start=start)],
            )
        df = pd.concat(user_preds)
        return df


    @staticmethod
    def predict_for_user(user, watched, pred, items, k):
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
''''''
