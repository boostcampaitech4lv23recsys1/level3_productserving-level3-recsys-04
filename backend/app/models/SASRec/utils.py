import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    """_summary_ : 유저가 안본 아이템 랜덤 샘플링 해주는 함수.
    Args:
        item_set (set): 유저의 아이템 set
        item_size (int): args.item_size(max_item + 2)
    Returns:
        item(int): item_id (item_set에 없는)  
    """    
    # 1 <= item < item_size, 1부터 하는 이유? 0은 패딩이니깐.
    item = random.randint(1, item_size - 2) # 1, item_size - 1
    # item_set 안에 없는 item이 나올 때 까지 계속 반복
    while item in item_set:
        item = random.randint(1, item_size - 2) # 1, item_size - 1
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose: # 메시지만 보내줍니다.
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    """
    Args:
        user_seq (2차원 list): [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        num_users (int): 유저 수
        num_items (int): 아이템 수(정확힌 max item_id)
    Returns:
        rating_matrix: 크기 (num_users, num_items) 유저-아이템 행렬, 유저의 마지막 2개 영화시청기록 뺌.
    """    
    row = [] # user_id가 담긴 리스트
    col = [] # 유저 별 item_id 리스트가 담긴 리스트
    data = [] # 1이 달린(positive sampling이라고 알려주는) 리스트.

    # user_id : 유저 번호, item_list : 해당 유저 item_id list
    for user_id, item_list in enumerate(user_seq): 
        for item in item_list[:-2]: # 해당 유저가 시청한 영화기록 마지막 2개를 제외함. 
            row.append(user_id)
            col.append(item)
            data.append(1)

    # 리스트를 넘파이 array로 바꿔줍니다.
    row = np.array(row)
    col = np.array(col) # 이 때 2차원 리스트는 겉만 np.array로 바뀌고 속은 list를 유지합니다.
    data = np.array(data)

    # 희소행렬 메트릭스 연산을 도와주는 scipy 내 csr_matrix 함수를 이용해 유저-아이템 행렬 제작합니다.
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    """
    Args:
        user_seq (2차원 list): [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        num_users (int): 유저 수
        num_items (int): 아이템 수(정확힌 max item_id)
    Returns:
        rating_matrix: 크기 (num_users, num_items) 유저-아이템 행렬, 유저의 마지막 1개 영화시청기록 뺌.
    """ 
    row = [] # user_id가 담긴 리스트
    col = [] # 유저 별 item_id 리스트가 담긴 리스트
    data = [] # 1이 달린(positive sampling이라고 알려주는) 리스트.

    # user_id : 유저 번호, item_list : 해당 유저 item_id list
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: # 해당 유저가 시청한 영화기록 마지막 1개를 제외함.
            row.append(user_id)
            col.append(item)
            data.append(1)

    # 리스트를 넘파이 array로 바꿔줍니다.
    row = np.array(row)
    col = np.array(col) # 이 때 2차원 리스트는 겉만 np.array로 바뀌고 속은 list를 유지합니다.
    data = np.array(data)

    # 희소행렬 메트릭스 연산을 도와주는 scipy 내 csr_matrix 함수를 이용해 유저-아이템 행렬 제작합니다.
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    """
    Args:
        user_seq (2차원 list): [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        num_users (int): 유저 수
        num_items (int): 아이템 수(정확힌 max item_id)
    Returns:
        rating_matrix: 크기 (num_users, num_items) 유저-아이템 행렬, 유저의 영화시청기록 빼지 않음.
    """ 
    row = [] # user_id가 담긴 리스트
    col = [] # 유저 별 item_id 리스트가 담긴 리스트
    data = [] # 1이 달린(positive sampling이라고 알려주는) 리스트.

    # user_id : 유저 번호, item_list : 해당 유저 item_id list
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]: # 해당 유저가 시청한 영화기록 제외하지 않고 모두 포함.
            row.append(user_id)
            col.append(item)
            data.append(1)

    # 리스트를 넘파이 array로 바꿔줍니다.
    row = np.array(row)
    col = np.array(col) # 이 때 2차원 리스트는 겉만 np.array로 바뀌고 속은 list를 유지합니다.
    data = np.array(data)

    # 희소행렬 메트릭스 연산을 도와주는 scipy 내 csr_matrix 함수를 이용해 유저-아이템 행렬 제작합니다.
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(data_file, preds):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "output/submission.csv", index=False
    )


def get_user_seqs(data_file):
    """
    Args:
        data_file : train_data 경로 

    Returns:
        user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열.
        => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        max_item : 가장 큰 item_id 
        valid_rating_matrix : 유저-아이템 희소행렬, 유저마다 마지막 2개 영화 시청기록은 뺌.(valid 위해.)
        test_rating_matrix : 유저-아이템 희소행렬, 유저마다 마지막 1개 영화 시청기록은 뺌.(test 위해.)
        submission_rating_matrix : 유저-아이템 희소행렬, 유저마다 영화 시청기록은 빼지 않음.
    """    
    # train 데이터 파일을 불러옵니다.
    rating_df = pd.read_csv(data_file)

    # lines : 유저인덱스/아이템리스트 형식의 판다스가 나옵니다.
    # ex) 11 [4643, 170, 531, 616, 2140, 2722, 2313, 2688, ...]
    lines = rating_df.groupby("user")["item"].apply(list)

    # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열.
    # ex) [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    user_seq = []
    item_set = set()
    for line in lines: # line : 한 유저의 아이템 리스트
        items = line
        user_seq.append(items) # append : 리스트를 하나의 원소로 보고 append함
        item_set = item_set | set(items) # | : 합집합 연산자
    # 기록된 가장 큰 아이템 id(번호)
    max_item = max(item_set)
    # len(lines) : 유저 수.
    num_users = len(lines)
    # num_items : 가장 큰 아이템 id를 기준으로 아이템 수 측정.(실제로는 훨신 작음.) 
    num_items = max_item + 2

    # valid_rating_matrix : 유저-아이템 희소행렬, 유저마다 마지막 1개 영화 시청기록은 뺌.(valid 위해.)
    valid_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    # submission_rating_matrix : 유저-아이템 희소행렬, 유저마다 영화 시청기록은 빼지 않음.
    submission_rating_matrix = generate_rating_matrix_submission(
        user_seq, num_users, num_items
    )
    return (
        user_seq,
        max_item,
        valid_rating_matrix,
        submission_rating_matrix,
    )


def get_user_seqs_long(data_file):
    """
    Args:
        data_file : train 데이터 파일 경로
    Returns:
        user_seq : 유저 id(번호) 순서대로 아이템 id 리스트 출력. 2차원 리스트(행길이는 유저 숫자)
        max_item : 가장 큰 아이템 id(번호).
        long_sequence : 아이템 id를 유저 id 순서대로 1차원으로 쭉 늘린 리스트.
    """    
    # train 데이터 파일을 불러옵니다.
    rating_df = pd.read_csv(data_file)

    # lines : 유저인덱스/아이템리스트 형식의 판다스가 나옵니다.
    # ex) 11 [4643, 170, 531, 616, 2140, 2722, 2313, 2688, ...]
    lines = rating_df.groupby("user")["item"].apply(list)

    # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열.
    # ex) [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    user_seq = []
    # long_sequence : 아이템 리스트를 1차원으로 늘려서 이어붙임. 
    long_sequence = []
    # item_set : 기록된 아이템 모두 담는 집합.
    item_set = set()
    for line in lines: # line : 한 유저의 아이템 리스트
        items = line
        long_sequence.extend(items) # extend : 리스트를 길게 이어붙임
        user_seq.append(items) # append : 리스트를 하나의 원소로 보고 append함
        item_set = item_set | set(items) # | : 합집합 연산자
    max_item = max(item_set) # 기록된 가장 큰 아이템 id(번호)

    return user_seq, max_item, long_sequence


def get_item2attribute_json(data_file):
    """
    Args:
        data_file : item과 genre의 mapping 데이터 파일 경로
    Returns:
        item2attribut : 딕셔너리, key => item id, value => genre의 list
        (ex : {
            "1":[8,12,13,5,9],
            "2":[8,13,9],
            "3":[5,6],
            ...
        })
    """
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    # item : 아이템 id, attributes : 해당 아이템 id의 genre 변수 리스트.
    for item, attributes in item2attribute.items(): 
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
