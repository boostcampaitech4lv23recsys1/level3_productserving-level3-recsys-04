def crawling2(area):
    import pandas as pd
    import numpy as np
    import re

    import warnings
    warnings.filterwarnings("ignore")

    num = 40

    path = f'/opt/ml/input/project/airflow/dags/area_csv/{area}/'
    lsts = []

    for i in range(1, num):
        try:
            lst = pd.read_csv(f'{path}rest_{i}.csv')
            lsts.append(lst)
        except: break
        
    data = lsts[0]

    for i in range(1, num):
        try: data = pd.concat([data,lsts[i]])
        except: break
        
    data['url']= data['url'].apply(lambda x: re.findall(r'/[0-9]+', x)[0][1:])
    data['review'] = data['review'].apply(lambda x: re.findall(r'[\d{1.2}]+', x))
    data['len'] = data['review'].apply(lambda x : len(x))
    data = data[data['len'] > 0]
    data['rating'] = data['review'].apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['count'] = data['review'].apply(lambda x: x[1] if len(x) == 2 else x[0])

    data['count'] = data['count'].astype('float')
    data = data.drop_duplicates('url') # restaurant
    data = data[data['count'] >= 10]

    data.to_csv(f'{path}/rest_concat.csv', index=False)