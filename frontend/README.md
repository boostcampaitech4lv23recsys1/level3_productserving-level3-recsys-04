## 개발 환경 (Dev)

1. node.js 설치
https://nodejs.org/ko/

2. yarn 설치
npm install -g yarn
npm install sweetalert2

3. 시작

3.1) yarn add global react-scripts

3.2) yarn start


## 배포 환경 (Prod)

Working Directory = "/home/ubuntu/code/frontend$"

1. `docker compose up -d`


### 프론트 컨테이너 삭제 후 재실행

1. `docker stop frontend`
2. `docker rm frontend`
3. `docker container run -t -p 3000:3000 front_end:latest `  (경로 frontend/ 안에서 실행

이렇게 나오면 실행 성공!
[+] Running 1/1
 ⠿ Container frontend  Started


 