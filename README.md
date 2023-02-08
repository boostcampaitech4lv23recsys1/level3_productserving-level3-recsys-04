
# **머글꺼니**

> **머글꺼니** : 오늘은 어떤 음식을 머글꺼니
> 최종 발표 [구글 슬라이드]() 및 [PDF]() & [Youtube]()

<br/>

------

## **프로젝트 동기**

대학내일연구소에 의하면 20대의 절반이상은 오늘 점심 메뉴를 고민하는 것을 가장 어려운 고민으로 뽑는데요.
메뉴 선택에 어려움을 겪는 젊은 세대를 위해 장소 내에서 강력하게 점심을 추천해주는 서비스을 기획했습니다.

<br/>

------

## 데이터 

- 식당 정보 및 유저 리뷰 정보: **[네이버 My Place](https://m.place.naver.com/my/feed)** 및 **[네이버](https://www.naver.com/)** 에서 웹 스크랩핑
- 식당 위치 정보 : **[Naver api](https://developers.naver.com/main/)** 지역 정보 검색 기능으로 식당 및 유저의 좌표 값 수집

- 서울시 내 총 **41460** 개의 식당, **382939** 명의 유저 데이터 수집

<br/>

------

## 모델

| 모델명                           | 참조                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| SASRec  | [Wang-Cheng Kang, Julian McAuley, Self-Attentive Sequential Recommendation,ICDM'18 ](https://arxiv.org/abs/1912.11160) |
| Multi-VAE | [Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara. 2018. Variational Autoencoders for Collaborative Filtering', WWW '18: Proceedings of the 2018 World Wide Web Conference](https://dl.acm.org/doi/10.1145/3178876.3186150) |
| EASE | [Harald Steck, Embarrassingly Shallow Autoencoders for Sparse Data, the Web Conference (WWW) 2019](https://arxiv.org/abs/1905.03375) |

<br/>

-----

## 오프라인 테스트

### Train Dataset

모델을 오프라인 테스트 하기 위해서 데이터를 train data, test data 로 나눠야하는데 이 때 두가지 방법으로 분할 하였습니다.
random 분할은 유저의 리뷰 중 무작위로 test data로 뽑아냈고
time 분할은 유저가 마지막으로 리뷰한 데이터를 test data로 뽑아냈습니다
![train data](https://user-images.githubusercontent.com/113089704/217433456-e54c2bf9-43db-4943-8d4a-d25faf559308.png)

<br/>

### 모델 별 성과

| 모델명| recall@20(Rand)|recall@20(Rand)|  Personalization    | 
| ----| ----| ----|----|
| SASRec| 5.65%| 5.96%|0.00669|
| MuiltiVAE| 11.23%| 10.02%|0.00253|
| EASE| 29.10%| 24.29%|0.00334|
|단순 인기도 모델 | 0.03%|0.03%|
|단순 랜덤 추천 | 0.01%|0.01%|

<br/>

 ------

## 프로젝트 구조

### 프로젝트 구조
![구조](https://user-images.githubusercontent.com/113089704/217406500-e15df2fb-d8f1-4a58-85fd-ea40cb8b77f0.png)

### 데이터 구조
![데이터 구조](https://user-images.githubusercontent.com/113089704/217406715-bb41ec55-655b-45a4-b6f8-8076f08a2362.png)

### airflow 
![airflow](https://user-images.githubusercontent.com/113089704/217407091-b1af9161-fdad-4d73-bfb8-dc8129fb9789.png)

### 폴더 구조
```bash
📦level3_productserving-level3-recsys-04
 ┣ 📂airflow
 ┃ ┣ 📂dags
 ┃ ┣ 📂ease
 ┃ ┣ 📂multi_vae
 ┃ ┣ 📂sasrec
 ┃ ┗ 📜batch_modeling.py
 ┣ 📂backend
 ┃ ┣ 📂app
 ┃ ┃ ┣ 📂models
 ┃ ┃ ┃ ┣ 📂data
 ┃ ┃ ┃ ┣ 📂ease
 ┃ ┃ ┃ ┣ 📂multivae
 ┃ ┃ ┃ ┗ 📂sasrec
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┣ 📜type.py
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┗ 📜__main__.py
 ┃ ┣ 📜README.md
 ┃ ┗ 📜requirements.txt
 ┣ 📂crawings
 ┃ ┣ 📂user_csv
 ┃ ┣ 📂area_csv
 ┃ ┣ 📜1.Kcrawling_rest_server.ipynb
 ┃ ┣ 📜2.Kcrwaling_concat.ipynb
 ┃ ┣ 📜3.Kcrawling_user_review.ipynb
 ┃ ┣ 📜3.Kcrawling_user_review.py
 ┃ ┣ 📜3_1.Kcrawling_user_review.py
 ┃ ┣ 📜4.Kcrawling_user_review_failed.ipynb
 ┃ ┣ 📜5.Kcrawling_get_rest_info.ipynb
 ┃ ┣ 📜chromedriver.exe
 ┃ ┣ 📜Kcsv_concat.ipynb
 ┃ ┗ 📜requirements.txt
 ┣ 📂database
 ┃ ┣ 📜DB_test.py
 ┃ ┣ 📜HowToUse.ipynb
 ┃ ┗ 📜HowToUse_local.ipynb
 ┣ 📂frontend
 ┃ ┣ 📂public
 ┃ ┃ ┣ 📂img
 ┃ ┃ ┗ 📜index.html
 ┃ ┣ 📂src
 ┃ ┃ ┣ 📂pages
 ┃ ┃ ┣ 📜index.js
 ┃ ┃ ┣ 📜setupProxy.js
 ┃ ┃ ┗ 📜style.css
 ┃ ┣ 📜README.md
 ┣ 📂model
 ┃ ┣ 📂cos_sim
 ┃ ┃ ┣ 📜cos_sim.ipynb
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📜rest.csv
 ┃ ┃ ┗ 📜user.csv
 ┃ ┣ 📂EASE
 ┃ ┃ ┣ 📜EASE.ipynb
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┗ 📜model.py
 ┃ ┣ 📂Multi-VAE
 ┃ ┃ ┣ 📜multi_vae.py
 ┃ ┃ ┗ 📜Multi_VAE_&_Multi_DAE.ipynb
 ┃ ┗ 📂sasrec
 ┃ ┃ ┣ 📜datasets.py
 ┃ ┃ ┣  ...
 ┃ ┃ ┗ 📜utils.py
 ┣ 📜.gitignore
 ┗ 📜README.md
```

------

## 팀원 소개

| <img src="https://user-images.githubusercontent.com/79916736/207600031-b46e76d2-cba3-4c94-9fc3-d9f29cd3bef8.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207600420-dd537303-d69d-439f-8cc8-5af648fe8941.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207601023-bbf9e64f-1447-41d8-991f-677593094592.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207600724-c140a102-39fc-4c03-8109-f214773a64fc.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/208005357-e98d106d-a207-4acd-ab4b-1abf7dbcb69f.png" width=200> | <img src="https://user-images.githubusercontent.com/65999962/210237522-72198783-f40c-491b-b8a7-6e6badf6cc24.jpg" width=200> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [김성연](https://github.com/KSY1526)                                            |                                           [배성재](https://github.com/SeongJaeBae)                                            |                                            [양승훈](https://github.com/Seunghoon-Schini-Yang)                                            |                                         [조수연](https://github.com/Suyeonnie)                                          |                                            [황선태](https://github.com/HSUNEH)                                            |                                            [홍재형](https://github.com/secrett2633)                                            |

### 팀 역할
- **김성연**: 모델링, 데이터베이스(SQLite), 데이터 전처리, metric 정의, airflow
- **배성재**: 데이터 크롤링, 프론트엔드(React), 서비스 배포, 식당 좌표 수집, MLflow
- **양승훈**: 모델링, 백엔드(FastAPI), 서비스 배포, MLflow, airflow
- **조수연**: 모델링, 백엔드(FastAPI), PPT
- **홍재형**: 데이터 크롤링, 데이터베이스(SQLite), 프론트엔드(React), 백엔드(FastAPI), airflow
- **황선태**: 모델링, 프론트엔드(React), 발표

<br/>

------

## Reference
- [Naver myplace](https://m.place.naver.com/my/feed)
- [Naver api](https://developers.naver.com/main/)

