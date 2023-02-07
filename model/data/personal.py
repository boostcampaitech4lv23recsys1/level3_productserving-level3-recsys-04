import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def personalizeion(pred_list):
    """ personalizeion score 뽑아주는 함수. 
    Args:
        pred_list (pandas): [index, [추천 목록 20개]] 

    Returns:
        score (int): 해당 추천의 personalizeion score
    """    
    #pred_list = pd.read_csv('../sasrec/SASRec_time_test_0130.csv')

    col = pred_list.columns[1]
    y_lst = []

    def fun(x):
        x = eval(x)
        y_lst.extend(x)
        return x

    pred_list[col] = pred_list[col].apply(lambda x : fun(x))
    y_array = np.array(y_lst)


    x_array = np.arange(pred_list.shape[0])

    for _ in range(19):
        x_array = np.vstack((x_array, np.arange(pred_list.shape[0])))

    x_array = x_array.T.reshape(-1)
    dat_array = np.ones(x_array.shape[0])
    matrix = sparse.csr_matrix((dat_array, (x_array, y_array)), shape = (pred_list.shape[0],y_array.max()+1), dtype=int)


    _sum = 0

    for i in (range(0, 39)):
        #print(i)
        idx = i * 10000
        a = cosine_similarity(matrix[idx:idx + 10000], matrix[idx:idx + 10000])
        a = np.triu(a, k = 1)
        _sum += a.sum()

    cnt = (pred_list.shape[0] / 10000) * (10000 * 10000 - 10000) // 2
    return _sum / cnt