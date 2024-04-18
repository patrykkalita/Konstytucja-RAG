FROM python:3.12

WORKDIR /weather
ADD . /weather

RUN pip install -r requirements.txt

CMD ["streamlit","run","main.py"]