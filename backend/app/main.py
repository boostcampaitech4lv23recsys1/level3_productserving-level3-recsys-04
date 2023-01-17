from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   
class Item(BaseModel):
    name: str
    location: str

    

@app.get("/")
def main_page():
    return {"message": "This is main page."}



data = [
    ["5c667add298eafd0547442d8", 1373442166],
    ["5c3737d3d764236c17947538", 1373442166],
    ["5c667add298eafd0547442d8", 1889162643]
]

users = {'5c667add298eafd0547442d8', '5c667add298eafd0547442d8'}

@app.get('/user/{user_id}')
def get_user_record(user_id: str):
    return [datum[1] for datum in data if datum[0] == user_id]


# @app.get('/signin/{user_id}')
# def user_signin(user_id: str):
#     if user_id in users:
#         return {"message": "success"}
#     else:
#         return {"message": "not registerd"}
@app.post('/signin/{user_id}')
def user_signin(user_id: str):
    if user_id in users:
        return {"message": "success"}
    else:
        return {"message": "not registerd"}

@app.post('/signin/')
def valid_signin(item: Item):
    return {"message": item}
    
