import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.cv.video_multi_object_tracking.utils.visualization import show_multi_object_tracking_result


model_id = 'damo/cv_yolov5_video-multi-object-tracking_fairmot'
video_multi_object_tracking = pipeline(Tasks.video_multi_object_tracking, model=model_id)

video_path = r'MOT17-03.mp4'

result = video_multi_object_tracking(video_path)
print('result is : ', result[OutputKeys.BOXES])
show_multi_object_tracking_result(video_path, result[OutputKeys.BOXES], result[OutputKeys.LABELS], "mot_res.avi")
