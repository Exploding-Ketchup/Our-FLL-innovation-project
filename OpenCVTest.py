import cv2
import time
from ProjectConstants import *

start_time = time.time()

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(1)

def detect_bounding_box(vid, saveFace=False):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        if saveFace:
            sub_face = vid[y:y+h, x:x+w]
            FaceFileName = "savedfaces/face_latest.jpg"
            cv2.imwrite(FaceFileName, sub_face)
            print("Timer Fired")
    return faces

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    saveFace = False
    current_time = time.time()
    if (current_time - start_time) > 5:
        #saveFace = True
        start_time = time.time()

    faces = detect_bounding_box(
        video_frame,
        saveFace=saveFace
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    capturedKey = cv2.waitKey(1)

    if capturedKey == ord("q"):
        break
    elif capturedKey == ord("s"):
        cv2.imwrite(SAVED_FACES_IMAGE, video_frame)
        print("Image Saved")
        time.sleep(3)
    else:
        pass

video_capture.release()
cv2.destroyAllWindows()