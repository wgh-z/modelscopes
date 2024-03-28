# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io

from flask import Flask, request
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


app = Flask(__name__)

DETECTION_URL = '/v1/mplug'


@app.route(DETECTION_URL, methods=['POST'])
def gsam():
    global pipeline_caption
    if request.method != 'POST':
        return

    if request.files.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im_data = Image.open(io.BytesIO(im_bytes))

        result = pipeline_caption(im_data)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing mPlug model')
    parser.add_argument('--port', default=2051, type=int, help='port number')
    opt = parser.parse_args()

    model_id = 'weight/damo/mplug_image-captioning_coco_base_zh'
    pipeline_caption = pipeline(Tasks.image_captioning, model=model_id)

    app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting with stat
