from pydantic import BaseModel
from typing import List


class RateRequest(BaseModel):
    user_id: int
    recipe_id: int
    rating: float


class SignInRequest(BaseModel):
    name: str
    location: str
    menu: str  # 1 : 식사, 2 : 카페/디저트


class SignInColdRequest(BaseModel):
    name: str
    location: str
    menu: str
    c1: bool
    c2: bool
    c3: bool
    c4: bool
    c5: bool
    c6: bool
    c7: bool
    c8: bool
    c9: bool


class AlbumRequest(BaseModel):
    user_id: str  # user
    rest_id: str  # url
    is_positive: bool
    model: str


class AlbumNegativeRequest(BaseModel):
    user_id1: str  # user
    rest_id1: str  # url
    is_positive1: bool
    model1: str
    user_id2: str  # user
    rest_id2: str  # url
    is_positive2: bool
    model2: str
    user_id3: str  # user
    rest_id3: str  # url
    is_positive3: bool
    model3: str


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
    model: str
    ment: str


class SignInResponse(BaseModel):
    state: str
    detail: str  # cold-start or not
    name: str
    restaurants0: List[Restaurant]  # rec 1
    restaurants1: List[Restaurant]  # rec 2
    restaurants2: List[Restaurant]  # rec 3


class SignInColdResponse(BaseModel):
    state: str
    detail: str  # cold-start or not


class AlbumResponse(BaseModel):
    user_id: str
    rest_id: str
    is_positive: bool


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
