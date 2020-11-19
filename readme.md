# Environment requirements

python3 modules:
tensorflow==2.3.0
opencv-python
matplotlib
flask
redis
celery

# RUN

## tfmodel

```python
python run_flask_server_yolov4tf.py
```

The model path is defined in line 193 ``saved_model_loaded = tf.saved_model.load('model/yolov4-416', tags=[tag_constants.SERVING])``.

## tflite model

```python
python run_flask_server_yolov4tflite.py
```

The model path is defined in line 53 ``interpreter,inputs,output_details = load_model_lite('model/yolov4-416-fp16.tflite')``.















