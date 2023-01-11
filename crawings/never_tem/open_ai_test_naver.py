import os
import sys
import urllib.request
client_id = "789Xk04GARJpb4omVvUq" # 개발자센터에서 발급받은 Client ID 값
client_secret = "oynUXBN1cW" # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote("송파구음식점")
url = "https://openapi.naver.com/v1/search/webkr?query=" + encText # JSON 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)