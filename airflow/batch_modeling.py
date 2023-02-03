import pandas as pd
import numpy as np
import pandas as pd
from numpy import random
import time
import sqlite3

import re

import random

import warnings
warnings.filterwarnings('ignore')

seed = 1998

def batch():
    """
    cold 유저를 포함한 모든 유저-아이템 로깅 있는 data_all 불러오기.
    """
    data = pd.read_csv('./data/data_all.csv')

    """
    Backend에서 사용하는 reccar_0130 DB 연결하기.
    """
    cnxn = sqlite3.connect("../backend/reccar_0130.db")
    cursor = cnxn.cursor()

    """
    DB 내 rest, user, positive, (nagetive) 불러오기.
    """
    select_sql = "select url, rest_code from rest"
    cursor.execute(select_sql)
    result = cursor.fetchall()
    rest = pd.DataFrame(result, columns = ['url', 'rest_code'])

    select_sql = "select user, user_code from user"
    cursor.execute(select_sql)
    result = cursor.fetchall()
    user = pd.DataFrame(result, columns = ['user', 'user_code'])

    select_sql = "select user, url from positive"
    cursor.execute(select_sql)
    result = cursor.fetchall()
    positive_df = pd.DataFrame(result, columns = ['userid', 'rest'])

    """
    positive 로깅 데이터 전처리 진행 (user_code, rest_code 넣어주기)
    """
    positive_df['date'] = '2023년 1월 31일' # 이건 매일 바꿔줘야.

    positive_df = pd.merge(positive_df, rest, left_on = 'rest', right_on = 'url', how = 'left')
    positive_df = positive_df[~positive_df['rest_code'].isnull()] # 음식점 DB에 없는 곳 제외.
    positive_df['rest_code'] = positive_df['rest_code'].astype('int')

    positive_df = pd.merge(positive_df, user, left_on = 'userid', right_on = 'user', how = 'left')
    positive_df['user_code'] = positive_df['user_code'].fillna(-1) # cold 유저 -1 넣기.
    positive_df['user_code'] = positive_df['user_code'].astype('int')

    positive_df = positive_df.drop(['url','user'], axis = 1)


    """
    positive 로깅 데이터를 이용해 다음 작업 진행.
    1. 모든 positive 기록 Data_all 내 저장
    2. (Cold 탈출 유저에 대해) 음식 테이블 내 cnt(식당 방문 횟수) 증가.
    3. (Cold 탈출 유저에 대해) 유저 테이블 내 새로운 유저로 등록.
    4. (Cold 탈출 유저를 포함해서) 새로운 Train/Test Data Set 구축.
    """

    # 1. 모든 positive 기록 Data_all 내 저장
    data = pd.concat([data, positive_df])
    data = data.drop_duplicates(['userid','rest'])
    max_user_idx = data['user_code'].max()

    # cold 탈출 유저 list 정의
    cold_cnt = data[data['user_code'] == -1]['userid'].value_counts()
    cold_not_user = list(cold_cnt[cold_cnt >= 6].index)

    # 2. (Cold 탈출 유저에 대해) 음식 테이블 내 cnt(식당 방문 횟수) 증가.
    rest_lst = []
    for userid in cold_not_user:
        rest_lst.extend(data[data['userid'] == userid]['rest_code'].values)

    for rest_code in rest_lst:
        select_sql = f"update rest set cnt = cnt + 1 where rest_code = {rest_code}"
        cursor.execute(select_sql)

    # 3. (Cold 탈출 유저에 대해) 유저 테이블 내 새로운 유저로 등록.
    idx = max_user_idx + 1
    if cold_not_user:
        new_user_df = pd.DataFrame()
        for userid in cold_not_user:
            data['user_code'][data['userid'] == cold_not_user[0]] = idx
            tmp_df = data[data['userid'] == cold_not_user[0]]
            tmp_df = tmp_df.groupby('user_code')['rest_code'].unique().to_frame().reset_index()
            tmp_df['user'] = userid
            new_user_df = pd.concat([new_user_df, tmp_df])
            idx += 1

        new_user_df['rest_code'] = new_user_df['rest_code'].apply(lambda x : list(x))
        new_user_df['rest_code'] = new_user_df['rest_code'].astype('str')

        a = new_user_df.values.tolist()
        for i in range(len(a)):
            a[i] = tuple(list(a[i]))

        for i in a:
            cursor.execute("INSERT INTO user VALUES (?, ?, ?)", i)
        cnxn.commit()

    ## cold 유저를 포함한 새로운 data_all 저장
    data = data[['userid', 'rest', 'user_code', 'rest_code', 'date']]
    data.to_csv('./data/data_all.csv', index = False)


    # 4. (Cold 탈출 유저를 포함해서) 새로운 Train/Test Data Set 구축.
    data = data[data['user_code'] != -1]
    data = data.reset_index(drop = True)

    _user = data['userid'].value_counts().reset_index()
    _user.columns = ['userid', 'cnt']
    data = pd.merge(data, _user, how = 'left', on = 'userid')

    data['point'] = data['cnt'] // 10 + 2
    data['tem'] = 1

    ## 시계열을 고려한 Test_set
    data = data.sort_values(['user_code', 'date'], ascending= [True, False])
    data['seq'] = data.groupby('user_code')['tem'].apply(lambda x : x.cumsum())
    train_time = data[data['seq'] > data['point']]
    test_time = data[data['seq'] <= data['point']]

    ## 랜덤하게 한 값을 뚫은 Test_set
    random.seed(seed)
    data['rand'] = data['rest'].apply(lambda x : random.random())
    data = data.sort_values(['user_code', 'rand'], ascending= True)
    data['seq'] = data.groupby('user_code')['tem'].apply(lambda x : x.cumsum())

    train_rand = data[data['seq'] > data['point']]
    test_rand = data[data['seq'] <= data['point']]

    train_rand = train_rand.sort_values(['user_code', 'date'], ascending= True)
    train_time = train_time.sort_values(['user_code', 'date'], ascending= True)
    data = data.sort_values(['user_code', 'date'], ascending= True)

    """
    새로운 Data csv 저장
    data_all : cold 유저를 저장(이전에 진행)
    train_rand : 유저별로 랜덤하게 test set 구축.
    train_time : 유저별로 마지막 아이템을 기준으로 test set 구축.
    """
    train_rand = train_rand[['userid', 'rest', 'user_code', 'rest_code', 'date']]
    train_time = train_time[['userid', 'rest', 'user_code', 'rest_code', 'date']]

    test_rand = test_rand[['userid', 'rest', 'user_code', 'rest_code']]
    test_time = test_time[['userid', 'rest', 'user_code', 'rest_code']]

    train_rand.to_csv('./data/train_rand.csv', index = False)
    test_rand.to_csv('./data/test_rand.csv', index = False)

    train_time.to_csv('./data/train_time.csv', index = False)
    test_time.to_csv('./data/test_time.csv', index = False)


    """
    모델링 후 Score와 .pt 파일 생성하기.
    """
    import sys
    sys.path.append('sasrec')
    sys.path.append('ease')

    from sasrec.main import sasrec_main
    from ease.main import ease_main

    sasrec_recall_score, sasrec_per_score = sasrec_main()
    ease_recall_score = ease_main()
    print("sasrec : ", sasrec_recall_score)
    print("ease : ", ease_recall_score)

    return


if __name__ == '__main__':
    batch()
