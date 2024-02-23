# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# import cv2


# model_id = 'damo/cv_tinynas_object-detection_damoyolo_safety-helmet'
# safety_hat_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

# name = {'0': 'safety hat', '1': 'no safety hat'}
# input_location = 'image_safetyhat.jpg'

# rtsp = 'rtsp://192.168.31.190:8554/stream'

# cap = cv2.VideoCapture(rtsp)

# while True:
#     ret, frame = cap.read()
#     if ret:
#         result = safety_hat_detection(frame)
#         scores = result['scores']
#         labels = result['labels']
#         boxes = result['boxes']
#         for score, label, box in zip(scores, labels, boxes):
#             print("score: ", score, "label: ", label, "box: ", box)
#     else:
#         break

# # im = cv2.imread(input_location)
# # result = safety_hat_detection(im)

# # scores = result['scores']
# # labels = result['labels']
# # boxes = result['boxes']

# # for score, label, box in zip(scores, labels, boxes):
# #     print("score: ", score, "label: ", label, "box: ", box)


from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.cv.video_multi_object_tracking.utils.visualization import show_multi_object_tracking_result


model_id = '../weight/damo/cv_yolov5_video-multi-object-tracking_fairmot'
video_multi_object_tracking = pipeline(Tasks.video_multi_object_tracking, model=model_id)
video_path = r'E:\Projects\test_data\test_video\MOT17-03-partial.mp4'
result = video_multi_object_tracking(video_path)
print('result is : ', result[OutputKeys.BOXES])
show_multi_object_tracking_result(video_path, result[OutputKeys.BOXES], result[OutputKeys.LABELS], "mot_res.avi")
