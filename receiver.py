import json
import os
from celery import Celery
import requests
import logging

logging.basicConfig(level=logging.INFO)

#app = Celery('send_result_tasks', broker='redis://localhost:6379/0')
REDIS_HOST_PORT = os.getenv('REDIS_HOST_PORT', "localhost:6379")

app = Celery('send_result_tasks', broker='redis://' + REDIS_HOST_PORT + '/0')

#TASK_ID = os.getenv('TASKID', 'IMGCAP_0001')
ORION_HOST_PORT = os.getenv("ORION_HOST_PORT", "192.168.12.222:30458")


@app.task(name="result_publish")
def send_result(result):
    logging.info("publishing result ({})...".format(result))
    build_ngsi_request(result)


def build_ngsi_request(result):
    orion_url = "http://{}/v2/op/update".format(ORION_HOST_PORT)
    logging.info("the orion url is: {}".format(orion_url))
    payload = {
        "actionType": "append",
        "entities": [
            {
#                "id": TASK_ID,
  "id": str(result).split(" @@ ")[0],
                "type": "Task",
                "result": {
#      "value": str(result),
                    "value": str(result).split(" @@ ")[1],
                    "type": "String"
                }
            }
        ]
    }
    headers = {'content-type': 'application/json'}
    r = requests.post(orion_url, data=json.dumps(payload), headers=headers)
    print(r.status_code)


if __name__ == "__main__":
    build_ngsi_request("haha")