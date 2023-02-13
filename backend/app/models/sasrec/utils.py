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
        item_set (list): 유저의 아이템 리스트
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


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


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


def get_user_seqs(train, test):
    """
    Args:
        train : train 데이터 user/item csv 파일.
        test : test t데이터 user/item csv 파일.

    Returns:
        user_seq : train 유저마다 따로 아이템 리스트 저장. 2차원 배열.
        => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        test_user_seq : test 유저마다 따로 아이템 리스트 저장. 2차원 배열.
        => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        max_item : 가장 큰 item_id, item의 개수를 나타냄.
        train_matrix : 유저-아이템 희소행렬
    """    

    # lines : 유저인덱스/아이템리스트 형식의 판다스가 나옵니다.
    # ex) 11 [4643, 170, 531, 616, 2140, 2722, 2313, 2688, ...]
    train_lines = train.groupby("user")["item"].apply(list)
    test_lines = test.groupby("user")["item"].apply(list)

    # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열.
    # ex) [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    user_seq = []
    test_user_seq = []
    
    item_set = set()

    for line in train_lines: # line : 한 유저의 아이템 리스트
        items = line
        user_seq.append(items) # append : 리스트를 하나의 원소로 보고 append함
        item_set = item_set | set(items) # | : 합집합 연산자

    for line in test_lines: # line : 한 유저의 아이템 리스트
        items = line
        test_user_seq.append(items) # append : 리스트를 하나의 원소로 보고 append함

    # 기록된 가장 큰 아이템 id(번호)
    max_item = max(item_set)
    # len(lines) : 유저 수.
    num_users = len(train_lines)
    # num_items : 가장 큰 아이템 id를 기준으로 아이템 수 측정
    num_items = max_item + 2


    # train_matrix : 유저-아이템 희소행렬
    train_matrix = generate_rating_matrix_submission(
        user_seq, num_users, num_items
    )
    return (
        user_seq,
        test_user_seq,
        max_item,
        train_matrix,
    )


def get_test_list(test):
    """
    Args:
    test_csv : 
    user, item
    0, 0
    0, 1
    1, 3
    1, 4
    ...
    
    Returns:
    user, item
    0, [0, 1]
    1, [3, 4, ..]
    ...
    """    
    # test id, 해당 유저가 방문한 rest item 정보를 담는 것 제작
    test = test.groupby('user')['item'].unique().to_frame().reset_index()
    return test


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


def recallk(_list, k = 3):
    """ label과 prediction 사이의 recall 평가 함수 
    Args:
        actual : 실제로 본 상품 리스트
        pred : 예측한 상품 리스트
        k : 상위 몇개의 데이터를 볼지 (ex : k=5 상위 5개의 상품만 봄)
    Returns: 
        recall_k : recall@k 
    """ 
    actual = _list[0]
    predicted = _list[1]
    set_actual = set(actual)
    recall_k = len(set_actual & set(predicted[:k])) / min(k, len(set_actual))
    return recall_k
