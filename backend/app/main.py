from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from .model import trash_model

# from app.routes import index, auth

from dataclasses import asdict
from PIL import Image
import torch


import mysql.connector
from mysql.connector.constants import ClientFlag
​

app = FastAPI()

origins = ["*"]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


config = {
    'user': 'root',
    'password': 'wogud1028',
    'host': '34.64.202.234',
    'client_flags': [ClientFlag.SSL],
    'ssl_ca': '/opt/ml/input/app/ssl/server-ca.pem',
    'ssl_cert': '/opt/ml/input/app/ssl/client-cert.pem',
    'ssl_key': '/opt/ml/input/app/ssl/client-key.pem'
}

config['database'] = 'rest'  # add new database to config dict
cnxn = mysql.connector.connect(**config)
cursor = cnxn.cursor()

select_sql = "select * from rest"# where rating = 4.42"
cursor.execute(select_sql)
result = cursor.fetchall()

_id, _x, _y,_cat,_name,_imgurl = zip(*result)


restaurants = result


@app.get("/")
def main_page():
    return {"message": "This is main page."}


class Restaurant(BaseModel):
    id: str
    x: int
    y: int
    name: str
    img_url: str


    def show_image(self):
        img = Image.open(self.img_url)
        img.show()


class User(BaseModel):
    id: str
    # nickname: str
    restaurants : List[Restaurant] = Field(default_factory=list)


    def add_restaurant(self, restaurant: Restaurant):
        if restaurant.id in [existing_id for existing_id in self.restaurants]:
            return self

        self.restaurants.append(restaurant)
        return self


data = [
    ["5c667add298eafd0547442d8", 1373442166],
    ["5c3737d3d764236c17947538", 1373442166],
    ["5c667add298eafd0547442d8", 1889162643]
]


@app.get('/user/{user_id}')
def get_user_record(user_id: str):
    return [datum[1] for datum in data if datum[0] == user_id]


@app.get('/restaurant/', description="식당 리스트를 가져옵니다")
async def get_restaurants() -> List:
    return restaurants


def get_restaurant_by_id(user_id: str) -> Optional[User]:
    return next((restaurant for restaurant in restaurants if restaurant.id == user_id), None)


def get_restaurant_image_by_id(rest_id: str) -> Optional[Restaurant]:
    rest = Restaurant(id=rest_id)
    rest.show_image()


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


@app.patch('/user/{user_id}', description="새로 리뷰를 등록한 식당을 업데이트합니다")
async def update_restaurant(user_id: str, restaurant_update: RestaurantUpdate):
    updated_restaurant = update_restaurant_by_id(user_id=user_id, restaurant_update=restaurant_update)

    if not updated_restaurant:
        return {"message":"리뷰 정보를 찾을 수 없습니다"}
    
    return updated_restaurant


class Prediction(User):
    name: str = 'predict_result'
    result


@app.get('/predict/{user_id}/{rest_id}', description="해당 유저의 정보를 모델에게 전달하고 예측 결과를 가져옵니다")
async def make_prediction(user_id: str, rest_id: str, model=trash_model):
    predict_result = model()
    prediction = Prediction(result=predict_result)
    return prediction



class Item(BaseModel):
    name: str


users = ['5c667add298eafd0547442d8', '5c3737d3d764236c17947538']

@app.post('/signin/{user_id}')
def user_signin(user_id: str):
    if user_id in users:
        return {"message": "success"}
    else:
        return {"message": "not registerd"}
