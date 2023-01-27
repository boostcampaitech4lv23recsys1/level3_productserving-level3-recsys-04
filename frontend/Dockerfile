FROM node:18.13.0-alpine3.16
WORKDIR /user/src/app
COPY ./package.json ./
RUN yarn add global react-scripts
COPY ./src ./src
COPY ./public ./public
ENTRYPOINT [ "yarn", "start" ]
