from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'weight/damo/mplug_image-captioning_coco_base_zh'
input_caption = 'demo1.jpg'

pipeline_caption = pipeline(Tasks.image_captioning, model=model_id)
result = pipeline_caption(input_caption)
print(result)
