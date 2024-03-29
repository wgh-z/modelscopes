#模型下载
from modelscope import snapshot_download


# model_dir = snapshot_download('damo/cv_tinynas_head-detection_damoyolo')  # 人头检测（计数）
# model_dir = snapshot_download('damo/cv_tinynas_object-detection_damoyolo_facemask')  # 口罩检测
# model_dir = snapshot_download('damo/cv_tinynas_object-detection_damoyolo_safety-helmet')  # 安全帽检测
# model_dir = snapshot_download('damo/cv_tinynas_object-detection_damoyolo_cigarette')  # 抽烟检测
# model_dir = snapshot_download('damo/cv_tinynas_object-detection_damoyolo_phone')  # 手机检测
# model_dir = snapshot_download('damo/cv_tinynas_object-detection_damoyolo_traffic_sign')  # 交通标志检测

# model_dir = snapshot_download('damo/cv_resnet50_face-detection_retinaface')  # 人脸关键点检测
# model_dir = snapshot_download('damo/cv_yolox-pai_hand-detection')  # 人手识别

# model_dir = snapshot_download('damo/cv_resnet18_card_correction')  # 票证检测矫正
# model_dir = snapshot_download('damo/cv_ResNetC3D_action-detection_detection2d')  # 日常动作检测（举手、吃喝、吸烟、打电话、玩手机、趴桌睡觉、跌倒、洗手、拍照）

# model_dir = snapshot_download('damo/cv_yolov5_video-multi-object-tracking_fairmot', cache_dir='./weight')  # 人脸关键点检测

# model_dir = snapshot_download('damo/nlp_csanmt_translation_en2zh_base', cache_dir='./weight', revision='v1.0.1')  # 英译中

# model_dir = snapshot_download('/damo/mplug_image-captioning_coco_base_zh', , cache_dir='./weight')  # 图像描述

model_dir = snapshot_download('iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx')  # 语音识别