from numpy import random
import pandas as pd

import random
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
    
path = '../data/'

train = pd.read_csv(path + 'G_test.csv')



def encoding(data):
    # 모델 시작 전 인코딩하는 과정입니다.
    user2idx = {user:idx for idx, user in enumerate(data['userid'].unique())}
    idx2user = {idx:user for idx, user in enumerate(data['userid'].unique())}

    ## 패딩 때문에 0 비워놓음
    item2idx = {item:(idx+1) for idx, item in enumerate(data['rest'].unique())}
    idx2item = {(idx+1):item for idx, item in enumerate(data['rest'].unique())}

    data_new = data.copy()

    data_new['userid'] = data_new['userid'].map(user2idx)
    data_new['rest'] = data_new['rest'].map(item2idx)

    # mapping 정보를 pickle 파일로 저장합니다. (backend의 inference 부분에 넘겨줘야 함)
    with open('../data/mapping.p', 'wb') as f:
        pickle.dump(user2idx, f)
        pickle.dump(idx2user, f)
        pickle.dump(item2idx, f)
        pickle.dump(idx2item, f)

    return data_new


# 모델 학습에서는 디코딩 안해도 될듯?
def decoding(output, idx2user , idx2item):

    # 모델 학습 후 디코딩하는 과정입니다.
    out_path = './output'
    # output = pd.read_csv(os.path.join(out_path, 'submission.csv'))
    output['user'] = output['user'].map(idx2user)
    output['item'] = output['item'].map(idx2item)
    output.to_csv(os.path.join(out_path, 'deco_trainAG_submission.csv'), index = False)
    
    return output



#####앙상블#####  앙상블은 sasrec 폴더 안에 말고, model 폴더 밑에 따로 만들어야 할듯?
def ensemble(out_path, baseline_csv, model_csv):
    # 베이스라인 저장된 csv 호출
    baseline = pd.read_csv(os.path.join(out_path, baseline_csv))
    # 모델 저장된 csv 호출
    model = pd.read_csv(os.path.join(out_path, model_csv))

    # tem : seq 만들어주기 위한 도구
    # seq : 랭킹을 매기기 위한 값. 낮을 수록 더 유망한 것.
    baseline['tem'] = 2
    model['tem'] = 2
    baseline['seq'] = baseline.groupby('user')['tem'].apply(lambda x : x.cumsum())
    model['seq'] = model.groupby('user')['tem'].apply(lambda x : x.cumsum())
    model['seq'] = model['seq'] - 1 # model를 한 단계 높게 처주기 위해.
    baseline['seq'] = baseline['seq'] # (+ 2) : 10개 중 baseline 4개만 반영하기 위해 +2, +4 등 조치 취함.

        # 베이스라인과 model 합침.
    final = pd.concat([baseline, model])
    # 베이스라인과 model에서 모두 추천하는 영화 찾기 위한 코드.
    # 두 모델에서 모두 추천하는 영화는 0순위로 놓기로 함
    final['seq'][final.duplicated(['user','item'], False)] = 0 # 0 : 0순위.
    # 이후 중복 제거
    final = final.drop_duplicates(['user','item'])
    # 유저 단위로, seq가 낮을 수록 더 높은 순위에 추천이기 때문에 이렇게 함.
    final = final.sort_values(['user','seq']).reset_index(drop = True)
    # 상위 10개만 추림
    final = final.groupby('user').apply(lambda x : x[:10]).reset_index(drop = True)
    final[['user','item']].to_csv(os.path.join(out_path, 'ensemble2.csv'), index = False)


### 실행~~
train_new, user2idx, idx2user, item2idx, idx2item = encoding(train)
test_new, user2idx, idx2user, item2idx, idx2item = encoding(train)
train_new.to_csv(os.path.join(path, 'train_new.csv'), index = False)
test_new.to_csv(os.path.join(path, 'train_new.csv'), index = False)


def train_test_split(data):
    # train/ test 분할하고 csv로 저장

    data['rand'] = data['rest'].apply(lambda x : random.random())
    _user = data['userid'].value_counts().reset_index()
    _user.columns = ['userid', 'cnt']
    data = pd.merge(data, _user, how = 'left', on = 'userid')
    data = data[data['cnt'] > 5].reset_index(drop = True)
    data = data[~(data['userid'] == '')].reset_index(drop = True)
    data = data.sort_values(['userid', 'rand']).reset_index(drop = True)
    data['tem'] = 1
    data['seq'] = data.groupby('userid')['tem'].apply(lambda x : x.cumsum())

    train = data[data['tem'] + 1 < data['seq']]
    test = data[data['tem'] + 1 >= data['seq']]

    train[['userid', 'rest']].to_csv('../data/train_enc.csv', index = False)
    test[['userid', 'rest']].to_csv('../data/test_enc.csv', index = False)
