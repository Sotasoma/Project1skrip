import cv2
import numpy as np
import time
import glob
import os

# Load Yolo
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture('0')
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1.5, color, 3)

    vehicles_entering = {}
    # vehicles_elapsed_time = {}
    area = [(100, 210), (550, 210), (630, 350), (10, 350)]

    # result = cv2.pointPolygonTest(np.array(area, np.int32), (int(center_x), int(center_y)), False)
    # if result >=0:
    vehicles_entering[class_id] = time.time()
    print(vehicles_entering)

    # area = [(200, 130), (480, 230), (630, 350), (10, 350)]
    # area2 = [(670, 900), (670, 990), (930, 840), (980, 1120)]
    if class_id in vehicles_entering:
        result = cv2.pointPolygonTest(np.array(area, np.int32), (int(center_x), int(center_y)), False)
    # print (result)
        if result >= 0:
        # elapsed_time = time.time() - vehicles_entering[class_id]
        # if class_id not in vehicles_elapsed_time:
        #     vehicles_elapsed_time[class_id] = elapsed_time
        # if class_id in vehicles_elapsed_time:
        #     elapsed_time = vehicles_elapsed_time[class_id
        # distance = 20
        # a_speed_ms = distance / elapsed_time
        # a_speed_kh = a_speed_ms * 3.6
        # cv2.imshow(frame, "Danger.png", ((int) (frame.shape[1]/2 - 268/2), (int) (frame.shape[0]/2 - 36/2)))
            cv2.putText(frame, "!BAHAYA!", ((int) (frame.shape[1]/2 - 268/2), (int) (frame.shape[0]/2 - 36/2)), font, 5, (0, 0, 255), 3)
    # for area in [area1, area2]
   
    for i, area in enumerate([area]):
        if i==1:
            continue
        cv2.polylines(frame, [np.array(area, np.int32)], True, (15,220,10), 2)

                

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()