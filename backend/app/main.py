from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from starlette.responses import JSONResponse

from model import trash_model
from type import *

# from app.routes import index, auth

from fastapi.param_functions import Depends
from PIL import Image
import torch


import mysql.connector
from mysql.connector.constants import ClientFlag


app = FastAPI()

origins = ["*"]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


################ mysql database 설정
config = {
    'user': 'root',
    'password': 'wogud1028',
    'host': '34.64.202.234',
    'client_flags': [ClientFlag.SSL],
    # 아래 인증키 경로들은 각자 환경에 맞게 수정 (언제 한번 통일 ㄱㄱ)
    #'ssl_ca': '/Users/hwang/AI_Tech_Frontend/level3_productserving-level3-recsys-04/db/ssl/server-ca.pem',
    #'ssl_cert': '/Users/hwang/AI_Tech_Frontend/level3_productserving-level3-recsys-04/db/ssl/client-cert.pem',
    #'ssl_key': '/Users/hwang/AI_Tech_Frontend/level3_productserving-level3-recsys-04/db/ssl/client-key.pem'
    'ssl_ca': r'C:\Users\bsj94\workspace\project\db\ssl\client-cert.pem',
    'ssl_cert': r'C:\Users\bsj94\workspace\project\db\ssl\client-cert.pem',
    'ssl_key': r'C:\Users\bsj94\workspace\project\db\ssl\client-key.pem'
}

config['database'] = 'rest'  # add "rest" database to config dict
cnxn = mysql.connector.connect(**config)
cursor = cnxn.cursor()
################



'''
to-do list

1. user 로그인
    - 해당 user가 유저 테이블에 존재하는 user인지 검증
2. user 로그인 후
    - user의 x, y 좌표 가져오기
    - user 방문 리스트 가져오기
    - 방문 리스트와 좌표 기준으로 식당 걸러내기
3. 식당 추천
    - 걸러낸 식당 모델에 넣고 결과 받기
    - Top 3 식당 데이터 반환
'''



@app.get("/")
def main_page():
    return {"message": "This is main page."}





def show_image(self):
    img = Image.open(self.img_url)
    img.show()



users = ['5c667add298eafd0547442d8', '5c3737d3d764236c17947538']


@app.post('/signin')
def signin(user: SignInRequest):
    if user.name in users:
        '''
        user.location으로 쿼리 날려서 좌표 가져오는 코드
        '''
        _x = 134; _y = 156
        
        
        '''
        모델 추천 결과 가져오는 코드
        '''
        rec_result = ["1675303081", "1867823297", "38969614", "1867823297", "1675303081"]
        cat1 = []; cat2 = []; cat3 = []
        for i, rest_id in enumerate(rec_result):
            # select_sql = f"select * from rest where id = {rest_id}"
            # cursor.execute(select_sql)
            # result = cursor.fetchall()[0]
            # result = result[:100]
            # restaurants = {i: dict(Restaurant(id=_id, x=_x, y=_y, tag=_tag, name=_name, img_url=_imgurl)) for i, (_id, _x, _y,_tag,_name,_imgurl) in enumerate(result)}
            # _id, _x, _y,_tag,_name,_imgurl = result
            # restaurant = Restaurant(id=_id, x=_x, y=_y, tag=_tag, name=_name, img_url=_imgurl)
            # restaurants.append(restaurant)
            restaurant_1 = Restaurant(id=i, x=_x, y=_y, tag='restaurant-tag', name='restaurant-name', img_url='imgurl')
            restaurant_2 = Restaurant(id=i+10, x=_x, y=_y, tag='restaurant-tag', name='restaurant-name', img_url='imgurl')
            restaurant_3 = Restaurant(id=i+20, x=_x, y=_y, tag='restaurant-tag', name='restaurant-name', img_url='imgurl')
            cat1.append(restaurant_1)
            cat2.append(restaurant_2)
            cat3.append(restaurant_3)
            

        return SignInResponse(
                state='start',
                detail=' not cold start',
                cat1 = cat1,
                cat2 = cat2,
                cat3 = cat3
            )
    return GeneralResponse(state='cold-start', detail='new user')


# 특정 식당 정보 가져오는 API
def get_restaurant(rest_id: str):
    select_sql = f"select * from rest where id = {rest_id}"
    cursor.execute(select_sql)
    result = cursor.fetchall()
    _id, _x, _y,_tag,_name,_imgurl = result[0]
    rest_info = dict(Restaurant(id=_id, x=_x, y=_y, tag=_tag, name=_name, img_url=_imgurl))
    rest_info['success'] = True
    return JSONResponse(rest_info)


def ml_model(user_id):
    """_summary_

    Args:
        user_id (_type_): 
    """
    return


class Prediction(BaseModel):
    name: str = 'predict_result'
    result: float

@app.get('/predict/{user_id}/{rest_id}', description="해당 유저의 정보를 모델에게 전달하고 예측 결과를 가져옵니다")
async def make_prediction(user_id: str, rest_id: str, model = trash_model()):
     predict_result = model(user_id, rest_id)
     prediction = Prediction(result=predict_result)
     return prediction


# 모든 식당 정보 가져오는 API
'''
@app.get('/restaurants/', description="모든 식당 리스트를 가져옵니다")
async def get_restaurants() -> List:
    select_sql = "select * from rest"# where rating = 4.42"
    cursor.execute(select_sql)
    result = cursor.fetchall()
    # _id, _x, _y,_cat,_name,_imgurl = zip(*result)
    restaurants = result
    return restaurants
'''





'''
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
'''


'''
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
'''
