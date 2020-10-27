import cv2

class local_cam():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
    def __iter__(self):
        while True:
            while True:
                ret, frame = self.cap.read()
                if ret:
                    break
            yield frame

if __name__ == "__main__":
    while True:
        try:
            for frame in local_cam():
                inputFrame = frame
                cv2.imshow('frame',inputFrame)
                if cv2.waitKey(1)==ord('q'):
                    break
        except:
            pass