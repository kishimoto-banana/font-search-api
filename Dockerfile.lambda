FROM python:3.9-slim

ARG APP_DIR="/home/app"
WORKDIR ${APP_DIR}
RUN pip install awslambdaric poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-dev

COPY ./models ./models
COPY ./app ./app

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "app/main.handler" ]
