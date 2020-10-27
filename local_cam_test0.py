import cv2
if __name__ == "__main__":

    cap = cv2.VideoCapture(0) #local camera
    while True:
        ret,frame=cap.read()
        if ret:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1)==ord('q'):
                break