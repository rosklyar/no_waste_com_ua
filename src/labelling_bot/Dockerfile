FROM python:3.9-slim-buster

RUN useradd --create-home appuser

WORKDIR /home/appuser/bot

RUN chown -R appuser:appuser /home/appuser/bot/

USER appuser

COPY ./requirements.txt /home/appuser/bot/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /home/appuser/bot/requirements.txt

COPY ./bot /home/appuser/bot

CMD ["python", "main.py"]