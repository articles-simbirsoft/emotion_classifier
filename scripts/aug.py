import cv2
import os
import numpy as np


# указываем путь к датасету
path = 'path'

for emo in os.listdir(path):
    print(emo, len(os.listdir(f"{path}//{emo}")))

# заполняем колличество сколько нужно добавить в каждый класс
d = {'neutral' : 2235, 'sad' :2370, 'surprised' : 4029}

for emo in os.listdir(path)[1:]:
    for img in os.listdir(f"{path}//{emo}"):
        path = os.path.join(f"{path}//", emo, img)
        img = cv2.imread(path)
        flipHorizontal = cv2.flip(img, 1)
        (h, w) = flipHorizontal.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), 25, 1.0)
        rotated = cv2.warpAffine(flipHorizontal, M, (w, h))
        cv2.imwrite(path[:-4] + "_new.png", rotated)
        d[emo] -= 1
        if d[emo] == 0:
            break


