# 구조도 

```bash
📦database
 ┣ 📂data
 ┃ ┣ 📜data.csv
 ┃ ┣ 📜data_all.csv
 ┃ ┣ 📜rest.csv
 ┃ ┣ 📜tag_tag.csv
 ┃ ┣ 📜test_rand.csv
 ┃ ┣ 📜test_time.csv
 ┃ ┣ 📜train_rand.csv
 ┃ ┣ 📜train_time.csv
 ┃ ┗ 📜user.csv
 ┣ 📂rest_csv
 ┃ ┣ 📜rest_concat_Dobong.csv
 ┃ ┣ 📜rest_concat_Dongdaemun.csv
 ┃ ┣ 📜rest_concat_Dongjag.csv
 ┃ ┗ 📜...(이하 26개 지역들)
 ┣ 📂user_csv
 ┃ ┣ 📜Dobong.csv
 ┃ ┣ 📜Dongdaemun.csv
 ┃ ┣ 📜Dongjag.csv
 ┃ ┗ 📜...(이하 26개 지역들)
 ┣ 📜data_EDA.ipynb
 ┣ 📜data_processing_DB.ipynb
 ┣ 📜data_processing_DataAll.ipynb
 ┣ 📜simple_model_recall.ipynb
 ┣ 📜personal.py
 ┗ 📜reccar_0130.db
```

## rest_csv, user_csv
크롤링을 통해 만들어진 식당과 유저 정보 csv가 구 별로 존재합니다.

## data_EDA.ipynb
data_all.csv 파일을 이용해 간단하게 데이터 EDA를 진행했습니다.

## data_processing_DB.ipynb
rest_csv, user_csv 폴더 내 csv 파일을 이용해 데이터를 전처리하였습니다.

data 폴더 내 data, user, rest, train, test 이름의 csv 파일을 만들었으며 DB도 구축했습니다.

## data_processing_DataAll.ipynb
Cold Start user도 포함된 data_all.csv 파일을 만들었습니다.

## simple_model_recall.ipynb
랜덤 추출, 단순 인기도 기반 추천의 recall 값을 구합니다.

## personal.py
얼마나 개인화 된 추천을 했는지 알아보는 personalization이라는 지표를 구하는 함수 입니다.

## tag_tag.csv
세세하게 나눠져 있는 소분류 태그를 총 10개의 대분류 태그로 변환하기 위해 사용하는 csv파일입니다.
