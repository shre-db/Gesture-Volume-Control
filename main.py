import cv2 as cv
import time
import numpy as np
import HT_module as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

w_cam, h_cam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
p_time = 0

detector = htm.HandDetector(detection_conf=0.7)
vol = 0
vol_bar = 400
vol_per = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lm_list = detector.findPosition(img, draw=False)
    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Finger Range (max:300, min: 50) -> Volume Range (max: 65, min:0)
        vol = np.interp(length, [50, 300], [min_vol, max_vol])
        vol_bar = np.interp(length, [50, 300], [400, 150])
        vol_per = np.interp(length, [50, 300], [0, 100])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)

    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(img, f'{int(vol_per)} %', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    cv.putText(img, f'FPS {int(fps)}', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv.imshow("Image", img)
    cv.waitKey(1)
