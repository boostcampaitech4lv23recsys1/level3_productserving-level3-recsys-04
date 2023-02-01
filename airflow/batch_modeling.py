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
    data = pd.read_csv('./data/data_all.csv')
    cnxn = sqlite3.connect("../backend/reccar_0130.db")
    cursor = cnxn.cursor()

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
    positive_df['date'] = '2023년 1월 31일' # 이건 매일 바꿔줘야.

    positive_df = pd.merge(positive_df, rest, left_on = 'rest', right_on = 'url', how = 'left')
    positive_df = positive_df[~positive_df['rest_code'].isnull()]
    positive_df['rest_code'] = positive_df['rest_code'].astype('int')

    positive_df = pd.merge(positive_df, user, left_on = 'userid', right_on = 'user', how = 'left')
    positive_df['user_code'] = positive_df['user_code'].fillna(-1)
    positive_df['user_code'] = positive_df['user_code'].astype('int')

    positive_df = positive_df.drop(['url','user'], axis = 1)

    data = pd.concat([data, positive_df])
    data = data.drop_duplicates(['userid','rest'])

    max_user_idx = data['user_code'].max()

    cold_cnt = data[data['user_code'] == -1]['userid'].value_counts()
    cold_not_user = list(cold_cnt[cold_cnt >= 6].index)

    rest_lst = []
    for userid in cold_not_user:
        rest_lst.extend(data[data['userid'] == userid]['rest_code'].values)

    for rest_code in rest_lst:
        select_sql = f"update rest set cnt = cnt + 1 where rest_code = {rest_code}"
        cursor.execute(select_sql)

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

    data = data[data['user_code'] != -1]
    data = data.reset_index(drop = True)

    _user = data['userid'].value_counts().reset_index()
    _user.columns = ['userid', 'cnt']
    data = pd.merge(data, _user, how = 'left', on = 'userid')

    data['point'] = data['cnt'] // 10 + 2
    data['tem'] = 1

    # 시계열을 고려한 Test_set
    data = data.sort_values(['user_code', 'date'], ascending= [True, False])
    data['seq'] = data.groupby('user_code')['tem'].apply(lambda x : x.cumsum())
    train_time = data[data['seq'] > data['point']]
    test_time = data[data['seq'] <= data['point']]

    # 랜덤하게 한 값을 뚫은 Test_set
    random.seed(seed)
    data['rand'] = data['rest'].apply(lambda x : random.random())
    data = data.sort_values(['user_code', 'rand'], ascending= True)
    data['seq'] = data.groupby('user_code')['tem'].apply(lambda x : x.cumsum())

    train_rand = data[data['seq'] > data['point']]
    test_rand = data[data['seq'] <= data['point']]

    train_rand = train_rand.sort_values(['user_code', 'date'], ascending= True)
    train_time = train_time.sort_values(['user_code', 'date'], ascending= True)
    data = data.sort_values(['user_code', 'date'], ascending= True)

    data[['userid', 'rest', 'user_code', 'rest_code', 'date']].to_csv('./data/data_all.csv', index = False)

    train_rand[['userid', 'rest', 'user_code', 'rest_code', 'date']].to_csv('./data/train_rand.csv', index = False)
    test_rand[['userid', 'rest', 'user_code', 'rest_code']].to_csv('./data/test_rand.csv', index = False)

    train_time[['userid', 'rest', 'user_code', 'rest_code', 'date']].to_csv('./data/train_time.csv', index = False)
    test_time[['userid', 'rest', 'user_code', 'rest_code']].to_csv('./data/test_time.csv', index = False)

    return