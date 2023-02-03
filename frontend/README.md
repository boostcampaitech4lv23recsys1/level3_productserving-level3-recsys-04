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

#### 프론트 이미지 & 컨테이너 삭제 후 재실행

1. `docker stop frontend`
2. `docker rm frontend`
3. `docker rmi nodejs`
4. `docker build -t nodejs .`
5. `docker container run --name frontend -t -p 3000:3000 nodejs`
