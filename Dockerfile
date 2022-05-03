FROM python:3.9-slim

ARG APP_DIR="/home/app"
WORKDIR ${APP_DIR}
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev --no-ansi
RUN poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

COPY ./models ./models
COPY ./app ./app

CMD [ "poetry", "run", "uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80" ]
