# 구조도

```bash
📦crawings
 ┣ 📂area_csv
 ┃ ┣ 📂GangJin
 ┃ ┃ ┣ 📜rest_concat_GangJin.csv
 ┃ ┣ 📂KongDong
 ┃ ┃ ┗ 📜rest_concat_KongDong.csv
 ┃ ┗ 📂...(이하 26개 지역들)
 ┣ 📂user_csv
 ┃ ┣ 📂Gangnam
 ┃ ┃ ┗ 📜user_concat_Gangnam.csv
 ┃ ┗ 📂KongDong
 ┃ ┃ ┣ 📜user_concat_KongDong.csv
 ┃ ┗ 📂...(이하 26개 지역들)
 ┣ 📜1.Kcrawling_rest_server.ipynb
 ┣ 📜2.Kcrwaling_concat.ipynb
 ┣ 📜3.Kcrawling_user_review.ipynb
 ┣ 📜4.Kcrawling_user_review_failed.ipynb
 ┣ 📜5.Kcrawling_get_rest_info.ipynb
 ┣ 📜Kcsv_concat.ipynb
 ┗ 📜requirements.txt
```
## area_csv
식당 정보를 저장해놓은 csv입니다. 구 별로 폴더가 나눠져 있으며 파일을 업로드하진 않았습니다.

## user_csv
유저의 식당 방문 기록을 저장해놓은 csv입니다. 구 별로 폴더가 나눠져 있으며 파일을 업로드하진 않았습니다.

## 1.Kcrawling_rest_server.ipynb
네이버 지도에서 음식점을 검색한 화면에서 음식점이름, 음식점 종류, 리뷰개수, 마이플레이스 링크를 가져옵니다.
## 2.Kcrwaling_concat.ipynb
Kcrawling_rest_server에서 가져온 데이터를 합치고 데이터를 가공합니다.
## 3.Kcrawling_user_review.ipynb
마이플레이스 링크에는 사용자가 남긴 리뷰와 음식점 정보가 있습니다.

사용자이름, 리뷰내용, 방문일자, 방문횟수, 음식점 대표 이미지를 수집합니다.
## 4.Kcrawling_user_review_failed.ipynb
마이플레이스 링크에서 사용자가 남긴 리뷰가 너무 많거나 인터넷 문제로 수집을 못 한 데이터를 수집합니다.
## 5.Kcrawling_get_rest_info.ipynb
마이플레이스 링크에서 식당의 대표 이미지와 x, y 좌표를 네이버 검색 api를 통해 수집합니다.
## Kcsv_concat.ipynb
수집한 파일을 합치고 중복을 제거해 줍니다.