from pydantic import BaseModel
from typing import List



class RateRequest(BaseModel):
    user_id: int
    recipe_id: int
    rating: float


class SignInRequest(BaseModel):
    name: str
    location: str


class GeneralRequest(BaseModel):
    qeury: str
    detail: str


class GeneralResponse(BaseModel):
    state: str
    detail: str

class Restaurant(BaseModel):
    id: str
    x: int
    y: int
    tag: str
    name: str
    img_url: str

class SignInResponse(BaseModel):
    state: str
    detail: str  # cold-start or not
    restaurants1 : List[Restaurant]
    restaurants2 : List[Restaurant]
    restaurants3 : List[Restaurant]

class User(BaseModel):
    name: str
    location: str




class ThemeSample(BaseModel):
    id: int
    title: str
    image: str


class ThemeSamples(BaseModel):
    theme_title: str
    theme_id: int
    samples: List[ThemeSample]


class ThemeListRequest(BaseModel):
    themes: List[int]


class ThemeListResponse(BaseModel):
    articles: List[ThemeSamples]


