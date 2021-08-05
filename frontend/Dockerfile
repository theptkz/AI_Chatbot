FROM node:16-alpine3.12 as builder

WORKDIR /usr/src/app/frontend
COPY ./package.json /usr/src/app/frontend
RUN apk add --no-cache g++ make python3 && npm install
# copy our react project
COPY . /usr/src/app/frontend

# perform npm build
ARG API_SERVER
ENV REACT_APP_API_SERVER=${API_SERVER}
RUN REACT_APP_API_SERVER=${API_SERVER} \ 
    npm run build

#########
# FINAL #
#########

# pull official base image
FROM node:16-alpine3.12

# set work directory
WORKDIR /usr/src/app/frontend

# install serve - deployment static server suggested by official create-react-app
RUN npm install -g serve

# copy our build files from our builder stage
COPY --from=builder /usr/src/app/frontend/build ./build
