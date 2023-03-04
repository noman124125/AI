# install  mediapipe
# >> pip install mediapipe

import cv2 as cv
import numpy as np
import mediapipe as mp
import time

frame_width = 680
frame_height = 460
cap = cv.VideoCapture(0)

# The "hands" module specifically is designed to detect and track human hand movements and gestures from a live video stream or a pre-recorded video.
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    resized_img = cv.resize(img, (frame_width, frame_height))

    img_rgb = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)

    # here we are processing hands
    results = hands.process(img_rgb)

    # here it will start detectiing hands
    # "results.multi_hand_landmarks" varible contains both hands
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        # here we seprating hands by looping
        # landmark contains only one hand
        for landmark in results.multi_hand_landmarks:
            # here we are drawing land marks on the image and making connecting the land marks
            mpdraw.draw_landmarks(resized_img, landmark,
                                  mphands.HAND_CONNECTIONS)

    cv.imshow("IMG", resized_img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
