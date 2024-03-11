import torch
import numpy as np

import cv2
import warnings
warnings.filterwarnings('ignore')


model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.eval()


def get_prediction(camera_img, confidence):
    pred = model([camera_img]).pandas().xyxy[0].to_numpy()
    if len(pred) > 0:
        people = pred[((pred[:, 6] == "person") & (np.float32(pred[:, 4]) > confidence))]
        pred_boxes = [[(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))] for i in people]
        return pred_boxes, len(people)
    return [], 0


def segment_instance(camera_img, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    boxes, person_num = get_prediction(camera_img, confidence)
    img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)
    if person_num > 0:
        for i in range(person_num):
            cv2.rectangle(img, 
                          tuple(map(int, boxes[i][0])), tuple(map(int, boxes[i][1])), 
                          color=(0, 255, 0), thickness=rect_th)
    return img


vid = cv2.VideoCapture(0) 
    
while True: 
    ret, frame = vid.read()
    cv2.imshow('frame', segment_instance(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release()
cv2.destroyAllWindows()


