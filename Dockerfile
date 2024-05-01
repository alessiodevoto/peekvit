FROM python:3.11.3

WORKDIR /peekvit

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install jupyterlab
RUN pip install notebook
RUN pip install ipykernel