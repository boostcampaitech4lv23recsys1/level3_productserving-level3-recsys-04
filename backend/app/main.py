from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from starlette.responses import JSONResponse

from type import *

from fastapi.param_functions import Depends
from PIL import Image
import torch

import sqlite3

from models.sasrec.inference import recommend


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
cnxn = sqlite3.connect("a.db", check_same_thread=False)
cursor = cnxn.cursor()
################ DB 설정 ################


@app.get("/")
def main_page():
    return {"message": "This is main page."}


def show_image(self):
    img = Image.open(self.img_url)
    img.show()


@app.post("/signin")
def signin(user: SignInRequest):
    select_sql = f"select * from user where user = '{user.name}'"
    cursor.execute(select_sql)
    result = cursor.fetchall()

    top_k = recommend(result[0][1])
    print(top_k)

    """
    user.location으로 쿼리 날려서 좌표 가져오는 코드
    """
    _x = 134
    _y = 156

    """
    모델 추천 결과 가져오는 코드
    """
    cat1 = []
    cat2 = []
    cat3 = []
    for i, rest_id in enumerate(top_k):
        restaurant_1 = Restaurant(
            id=i,
            x=_x,
            y=_y,
            tag="restaurant-tag",
            name=str(rest_id),
            img_url="imgurl",
        )
        restaurant_2 = Restaurant(
            id=i + 10,
            x=_x,
            y=_y,
            tag="restaurant-tag",
            name="restaurant-name",
            img_url="imgurl",
        )
        restaurant_3 = Restaurant(
            id=i + 20,
            x=_x,
            y=_y,
            tag="restaurant-tag",
            name="restaurant-name",
            img_url="imgurl",
        )
        cat1.append(restaurant_1)
        cat2.append(restaurant_2)
        cat3.append(restaurant_3)

    return SignInResponse(
        state='start',
        detail='not cold start',
        restaurants1 = cat1,
        restaurants2 = cat2,
        restaurants3 = cat3
    )


# 특정 식당 정보 가져오는 API
def get_restaurant(rest_id: str):
    select_sql = f"select * from rest where id = {rest_id}"
    cursor.execute(select_sql)
    result = cursor.fetchall()
    _id, _x, _y, _tag, _name, _imgurl = result[0]
    rest_info = dict(
        Restaurant(id=_id, x=_x, y=_y, tag=_tag, name=_name, img_url=_imgurl)
    )
    rest_info["success"] = True
    return JSONResponse(rest_info)


def ml_model(user_id):
    """_summary_

    Args:
        user_id (_type_):
    """
    return


class Prediction(BaseModel):
    name: str = "predict_result"
    result: float

# @app.get('/predict/{user_id}/{rest_id}', description="해당 유저의 정보를 모델에게 전달하고 예측 결과를 가져옵니다")
# async def make_prediction(user_id: str, rest_id: str, model = trash_model()):
#      predict_result = model(user_id, rest_id)
#      prediction = Prediction(result=predict_result)
#      return prediction


# 모든 식당 정보 가져오는 API
"""
@app.get('/restaurants/', description="모든 식당 리스트를 가져옵니다")
async def get_restaurants() -> List:
    select_sql = "select * from rest"# where rating = 4.42"
    cursor.execute(select_sql)
    result = cursor.fetchall()
    # _id, _x, _y,_cat,_name,_imgurl = zip(*result)
    restaurants = result
    return restaurants
"""


"""
def get_restaurant_image_by_id(rest_id: str) -> Optional[Restaurant]:
    rest = Restaurant(id=rest_id)
    rest.show_image()


def get_restaurant_by_id(user_id: str) -> Optional[User]:
    return next((restaurant for restaurant in restaurants if restaurant.id == user_id), None)


class RestaurantUpdate(BaseModel):
    restaurants: List[User] = Field(default_factory=list)


def update_restaurant_by_id(user_id: str, restaurant_update: RestaurantUpdate):
    existing_restaurant = get_restaurant_by_id(user_id=user_id)

    if not existing_restaurant:
        return
    else:
        updated_restaurant = existing_restaurant.copy()
        for next_restaurant in restaurant_update.restaurants:
            updated_restaurant.add_restaurant(next_restaurant)

        return updated_restaurant
"""


"""
@app.patch('/user/{user_id}', description="새로 리뷰를 등록한 식당을 업데이트합니다")
async def update_restaurant(user_id: str, restaurant_update: RestaurantUpdate):
    updated_restaurant = update_restaurant_by_id(user_id=user_id, restaurant_update=restaurant_update)

    if not updated_restaurant:
        return {"message":"리뷰 정보를 찾을 수 없습니다"}
    
    return updated_restaurant


class Prediction(User):
    name: str = 'predict_result'

@app.get('/predict/{user_id}/{rest_id}', description="해당 유저의 정보를 모델에게 전달하고 예측 결과를 가져옵니다")
async def make_prediction(user_id: str, rest_id: str, model=trash_model):
    predict_result = model()
    prediction = Prediction(result=predict_result)
    return prediction



class Item(BaseModel):
    name: str


users = ['5c667add298eafd0547442d8', '5c3737d3d764236c17947538']
"""
