from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from type import *

from PIL import Image
import sqlite3

from models.sasrec.inference import recommend as sasrec_inference
from models.ease.inference import recommend as ease_inference
from models.multivae.inference import recommend as multivae_inference
import urllib.request

import random
from bs4 import BeautifulSoup
import requests

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


################ DB 설정 ################
cnxn = sqlite3.connect("reccar_0202.db", check_same_thread=False)
cursor = cnxn.cursor()
################ DB 설정 ################


@app.get("/")
def main_page():
    return {"message": "This is main page."}


def show_image(self):
    img = Image.open(self.img_url)
    img.show()


@app.post("/api/album")
def album(data: AlbumRequest):
    print(data.model)
    if data.is_positive:
        cursor.executemany(
            "insert into positive values (?, ?, ?)",
            [(data.user_id, data.rest_id, data.model)],
        )
        select_sql = "select * from positive"
    else:
        cursor.executemany(
            "insert into negative values (?, ?, ?)",
            [(data.user_id, data.rest_id, data.model)],
        )
        select_sql = "select * from negative"
    cnxn.commit()
    cursor.execute(select_sql)
    result = cursor.fetchall()
    print(result)
    return AlbumResponse(
        user_id=data.user_id,
        rest_id=data.rest_id,
        is_positive=data.is_positive,
        model=data.model,
    )

@app.post("/api/album/negative")
def album(data: AlbumNegativeRequest):
    
    cursor.executemany(
        "insert into negative values (?, ?, ?)",
        [(data.user_id1, data.rest_id1, data.model1)],
    )
    cursor.executemany(
        "insert into negative values (?, ?, ?)",
        [(data.user_id2, data.rest_id2, data.model2)],
    )
    cursor.executemany(
        "insert into negative values (?, ?, ?)",
        [(data.user_id3, data.rest_id3, data.model3)],
    )
    select_sql = "select * from negative"
    cnxn.commit()
    cursor.execute(select_sql)
    result = cursor.fetchall()
    print(result)
    return AlbumResponse(
        user_id=data.user_id1,
        rest_id=data.rest_id1,
        is_positive=data.is_positive1,
        model=data.model1,
    )


@app.post("/api/signin")
def signin(user: SignInRequest):
    """
    user의 코드로 해당 user_list 받기
    """
    select_sql = f"select * from user where user = '{user.name}'"
    cursor.execute(select_sql)

    # user_list : [(user_code, rest_code, user)]
    user_list = cursor.fetchall()

    """
    전체 아이템의 크기 구하기.
    """
    select_sql = f"select max(rest_code) from rest"
    cursor.execute(select_sql)
    max_item = cursor.fetchall()  # [(41460,)]

    """
    user.location으로 쿼리 날려서 좌표 가져오는 코드
    """
    # 향후 user.location으로 x,y 받아야함.
    _x, _y = get_xy(user.location)  # _x = 314359, _y = 547462
    _inter = 1000  # 허용 가능한 거리, 임시방편.

    _input = (_x - _inter, _x + _inter, _y - _inter, _y + _inter, "음식아님", "카페&디저트")
    
    """
    user.name 쿼리 날려서 좌표 가져오는 코드
    """
    user_name = get_name(user.name)

    """
    모델을 이용한 Top3 추출
    """

    if not user_list:  # 만약 유저가 없는 사람이라면? 거리 내 인기도 기반 Top3 추천.
        return SignInColdResponse(
            state="start",
            detail="cold start",
        )

    else:
        if user.menu == "1":  # 식사인경우
            select_sql = "select DISTINCT rest_code from rest where ((x > ?) AND (x < ?) AND (y > ?) AND (y < ?) AND (tag != ?) AND (tag != ?))"
        else:  # 카페&디저트인 경우
            select_sql = "select DISTINCT rest_code from rest where ((x > ?) AND (x < ?) AND (y > ?) AND (y < ?) AND (tag != ?) AND (tag = ?))"

        cursor.execute(select_sql, _input)
        results = cursor.fetchall()
        rest_codes = [rest_code[0] for rest_code in results]
        if len(rest_codes) < 30:  # 식당이 30개보다 적다면 에러메세지
            return SignInColdResponse(
                state="start",
                detail="low data",
            )
        sasrec_top_k = sasrec_inference(user_list[0][1], rest_codes, max_item[0][0] - 1)
        ease_top_k = ease_inference(user_list[0][0], user_list[0][1], set(rest_codes))
        multivae_top_k = multivae_inference(rest_codes=user_list[0][1])
        select_sql += " order by cnt DESC"
        cursor.execute(select_sql, _input)
        results = cursor.fetchall()
        rulebase_top_k = [rest_code[0] for rest_code in results[:10]]
    print(sasrec_top_k)
    print(ease_top_k)
    print(multivae_top_k)
    print(rulebase_top_k)
    
    """
    모델 추천 결과 가져오는 코드
    """
    cat0 = []
    cat1 = []
    cat2 = []

    def add_top_k(model_top_k):
        for i, model_info in enumerate(model_top_k):
            rest_id, model_name = model_info
            ment = {
                "sasrec": "님의 최근 방문한 음식점을 고려한 추천입니다.",
                "multivae": "님의 숨겨진 취향을 고려한 추천입니다.",
                "ease": "님과 유사한 유저들을 고려한 추천입니다.",
                "rulebase": "님의 지역 내 음식점 인기도를 고려한 추천입니다.",
            }
            restaurant_1 = get_restaurant(rest_id, model_name, ment[model_name])
            if i % 3 == 0:
                cat0.append(restaurant_1)
            elif i % 3 == 1:
                cat1.append(restaurant_1)
            elif i % 3 == 2:
                cat2.append(restaurant_1)

    sasrec_top_k = [(top_k, "sasrec") for top_k in sasrec_top_k]    
    rulebase_top_k = [(top_k, "rulebase") for top_k in rulebase_top_k]
    ease_top_k = [(top_k, "ease") for top_k in ease_top_k]
    multivae_top_k = [(top_k, "multivae") for top_k in multivae_top_k]
    all_top_k = sasrec_top_k + rulebase_top_k + ease_top_k + multivae_top_k
    random.shuffle(all_top_k)
    add_top_k(all_top_k)
    
    return SignInResponse(
        state="start",
        detail="not cold start",
        name=str(user_name),
        restaurants0=cat0,  # rec 1
        restaurants1=cat1,  # rec 2
        restaurants2=cat2,  # rec 3
    )


@app.post("/api/signin/cold")
def signin(user: SignInColdRequest):
    """
    전체 아이템의 크기 구하기.
    """
    select_sql = f"select max(rest_code) from rest"
    cursor.execute(select_sql)

    """
    user.location으로 쿼리 날려서 좌표 가져오는 코드
    """
    # 향후 user.location으로 x,y 받아야함.
    _x, _y = get_xy(user.location)  # _x = 314359, _y = 547462
    _inter = 1000  # 허용 가능한 거리, 임시방편.
    _input = (_x - _inter, _x + _inter, _y - _inter, _y + _inter, "음식아님", "카페&디저트")

    """
    user.name 쿼리 날려서 좌표 가져오는 코드
    """
    user_name = get_name(user.name)

    """
    모델을 이용한 Top3 추출
    """
    tags = ""
    if user.c1: tags += " AND (tag != '한식')"
    if user.c2: tags += " AND (tag != '중식')"
    if user.c3: tags += " AND (tag != '일식')"
    if user.c4: tags += " AND (tag != '동남아음식')"
    if user.c5: tags += " AND (tag != '패스트푸드')"
    if user.c6: tags += " AND (tag != '고기')"
    if user.c7: tags += " AND (tag != '양식')"
    if user.c8: tags += " AND (tag != '해산물')"
    if user.c9: tags += " AND (tag != '분식&샐러드')"
    
    if user.menu == "1":  # 식사인 경우
        select_sql = f"select DISTINCT rest_code from rest where ((x > ?) AND (x < ?) AND (y > ?) AND (y < ?) AND (tag != ?) AND (tag != ?){tags}) order by cnt DESC"
    else:  # 카페&디저트인 경우
        select_sql = f"select DISTINCT rest_code from rest where ((x > ?) AND (x < ?) AND (y > ?) AND (y < ?) AND (tag != ?) AND (tag = ?){tags}) order by cnt DESC"
    
    cursor.execute(select_sql, _input)
    results = cursor.fetchall()
    top_k = [rest_code[0] for rest_code in results[:30]]
    print(top_k)

    """
    모델 추천 결과 가져오는 코드
    """
    cat0 = []
    cat1 = []
    cat2 = []
    for i, rest_id in enumerate(top_k):
        select_sql = f"select url, x, y, image, tag, name from rest where rest_code = {rest_id}.0"  # where rating = 4.42"
        cursor.execute(select_sql)
        url, x, y, image, tag, restaurant = cursor.fetchall()[0]

        restaurant_1 = Restaurant(
            id=url,
            x=x,
            y=y,
            tag=tag,
            name=restaurant,
            img_url=image,
            model="cold start",
            ment = "님의 지역 내 음식점 인기도를 고려한 추천입니다."
        )
        if i % 3 == 0:
            cat0.append(restaurant_1)
        elif i % 3 == 1:
            cat1.append(restaurant_1)
        elif i % 3 == 2:
            cat2.append(restaurant_1)

    return SignInResponse(
        state="start",
        detail="not cold start",
        name=str(user_name),
        restaurants0=cat0,  # rec 1
        restaurants1=cat1,  # rec 2
        restaurants2=cat2,  # rec 3
    )


def get_restaurant(rest_id, model_name, ment):
    """
    rest_id를 입력하면 화면에 띄울 Restaurant 클래스를 배출해주는 함수.
    """
    select_sql = (
        f"select url, x, y, image, tag, name from rest where rest_code = {rest_id}.0"
    )
    cursor.execute(select_sql)
    url, x, y, image, tag, restaurant = cursor.fetchall()[0]

    restaurant = Restaurant(
        id=url,
        x=x,
        y=y,
        tag=tag,
        name=restaurant,
        img_url=image,
        model=model_name,
        ment=ment,
    )

    return restaurant


def get_name(target: str):
    # target = '6130db4973adbe125329a3e4'
    url = "https://m.place.naver.com/my/{}/review?v=2".format(target)
    req = requests.get(url)
    soup = BeautifulSoup(req.content, "html.parser", from_encoding="cp949")
    id = soup.select_one('meta[property="article:author"]')["content"]
    return id


def get_xy(location: str):
    client_id = "789Xk04GARJpb4omVvUq"  # 개발자센터에서 발급받은 Client ID 값
    client_secret = "oynUXBN1cW"  # 개발자센터에서 발급받은 Client Secret 값
    encText = urllib.parse.quote(location)
    url = "https://openapi.naver.com/v1/search/local?query=" + encText  # JSON 결과
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        # print(response_body.decode('utf-8'))
        x = response_body.decode("utf-8").split('"')[-6]
        y = response_body.decode("utf-8").split('"')[-2]
        return int(x), int(y)
    else:
        # print("Error Code:" + rescode)
        return 0, 0


class Prediction(BaseModel):
    name: str = "predict_result"
    result: float
