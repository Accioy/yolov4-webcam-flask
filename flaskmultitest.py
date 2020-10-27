# coding=utf-8




import threading,time
from flask import Flask, render_template, Response


lock = threading.Lock()
loginfo = "1..."

print("2...")

inputFrame=None
outputFrame=None

app = Flask(__name__)

@app.route('/')
def index():
    global loginfo
    """Video streaming home page."""
    return render_template('index.html', loginfo = loginfo)


def get_frame():
    global inputFrame, loginfo
    while True:
        try:
            for i in range(100):
                inputFrame = i
        except Exception as e:
            loginfo = 'error, %s. \r\n\r\n Try again in %s seconds.' % (e, str(10))
            print(loginfo)
            loginfo = 'errorgetframe'
            time.sleep(10)
            pass


def run_detection(img,model):
    global outputFrame, lock, loginfo
    pred_bbox = 'rundetect'
    return pred_bbox


def object_detection(model):


    global inputFrame, outputFrame, lock, loginfo

    image_lists = []
    print("Begin to get video frames...")

    while True:
        try:
            image_lists.append(inputFrame)
            if len(image_lists) == 1:
                tmp = image_lists[0]+200

                with lock:
                    outputFrame = tmp

                r =  ' @@ ' + 'detect'
                time.sleep(3)
                image_lists = []

        except Exception as e:
            loginfo = 'error, %s. \r\n\r\n Try again in %s seconds.' % (e, str(3))
            print(loginfo)
            loginfo = ' @@ ' + loginfo

            time.sleep(3)
            pass


def generate():
    """Video streaming generator function."""
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # # encode the frame in JPEG format
            # (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            #
            # ensure the frame was successfully encoded
            # if not flag:
            #     continue
            encodedImage = outputFrame
        # yield the output frame in the byte format
        yield (encodedImage)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate())


if __name__ == "__main__":
    # start a thread that will get frames
    print("loading weights and engine file...")

    print("Weights and engine file loaded.")
    t1 = threading.Thread(target=get_frame)
    t1.daemon = True
    t1.start()

    # wait 3 seconds to get the frames ready
    time.sleep(3)

    # start a thread that will perform motion detection
    t = threading.Thread(target=object_detection,args = (1,))
    t.daemon = True
    t.start()
    app.run(host='127.0.0.1', threaded=True, debug=True, port=5000)
