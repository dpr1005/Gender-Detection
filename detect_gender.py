#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    detect_gender.py.py
# @Author:      Daniel Puente Ramírez
# @Time:        7/11/21 17:59

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv


def main():
    # Load the model
    model = load_model('gender_detection.model')

    # Open the webcam
    webcam = cv2.VideoCapture(0)
    classes = ['man', 'women']

    # Loop through frames
    while webcam.isOpened():
        # Read frame from webcam
        status, frame = webcam.read()

        # Apply face detection
        face, confidence = cv.detect_face(frame)

        # Loop through detected faces
        for index, fc in enumerate(face):
            # Get coordinates of the rectangle of minimum area around the face
            (startX, startY) = fc[0], fc[1]
            (endX, endY) = fc[2], fc[3]

            # Draw the rectangle over the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

                # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

            # get label with max accuracy
            idx = np.argmax(conf)
            label = classes[index]

            label = "{}: {:.2f}%".format(label, conf[index] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # display output
        cv2.imshow("gender detection", frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()