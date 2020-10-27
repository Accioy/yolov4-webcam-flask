from celery import Celery
import os

# app = Celery('send_result_tasks', broker='redis://localhost:6379/0')
REDIS_HOST_PORT = os.getenv('REDIS_HOST_PORT', "localhost:6379")

app = Celery('send_result_tasks', broker='redis://' + REDIS_HOST_PORT + '/0')


@app.task(name='result_publish')
def send_result(result):
    pass
