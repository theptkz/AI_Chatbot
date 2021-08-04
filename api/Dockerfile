FROM python:3.8.2 as builder

WORKDIR  /usr/src/app/api

# set environment variables 
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY ./requirements.txt  /usr/src/app/api
RUN pip wheel --no-cache-dir --no-deps --wheel-dir  /usr/src/app/api/wheels -r requirements.txt

#########
# FINAL #
#########

# pull official base image
FROM python:3.8.2


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    netcat &&\
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install dependencies
COPY --from=builder  /usr/src/app/api/wheels /wheels
COPY --from=builder  /usr/src/app/api/requirements.txt .
RUN pip install --no-cache /wheels/*

# set work directory
WORKDIR  /usr/src/app/api

# copy entrypoint.sh
COPY ./entrypoint.sh  /usr/src/app/api/entrypoint.sh

# copy our django project
COPY . .

# run entrypoint.sh
RUN chmod +x  /usr/src/app/api/entrypoint.sh
ENTRYPOINT ["/usr/src/app/api/entrypoint.sh"]
