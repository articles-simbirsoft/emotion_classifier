# import the opencv library
import cv2
from ultralytics import YOLO
import numpy as np
import os
# define a video capture object


# путь к файлу с шаблонами Хаара
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

# инициализировали каскадный классификатор
faceCascade = cv2.CascadeClassifier(cascPathface)

model = YOLO('../models/best.pt')


vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # преобразуем кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # инициализируем детектор
    faces = faceCascade.detectMultiScale(frame,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) != 0:
        (x, y, w, h) = faces[0]

        face = gray[y:y+h, x:x+w]

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (50, 50)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        results = model(face)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        print(names_dict[np.argmax(probs)])

        # Using cv2.putText() method
        image = cv2.putText(frame, f'{names_dict[np.argmax(probs)]}', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow('frame', image)

    else:
        cv2.imshow('frame', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
