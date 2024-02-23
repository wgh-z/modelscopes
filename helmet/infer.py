from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2


model_id = 'damo/cv_tinynas_object-detection_damoyolo_safety-helmet'
safety_hat_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

name = {'0': 'safety hat', '1': 'no safety hat'}
input_location = 'image_safetyhat.jpg'

rtsp = 'rtsp://192.168.31.190:8554/stream'

cap = cv2.VideoCapture(rtsp)

while True:
    ret, frame = cap.read()
    if ret:
        result = safety_hat_detection(frame)
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']
        for score, label, box in zip(scores, labels, boxes):
            print("score: ", score, "label: ", label, "box: ", box)
    else:
        break

# im = cv2.imread(input_location)
# result = safety_hat_detection(im)

# scores = result['scores']
# labels = result['labels']
# boxes = result['boxes']

# for score, label, box in zip(scores, labels, boxes):
#     print("score: ", score, "label: ", label, "box: ", box)
