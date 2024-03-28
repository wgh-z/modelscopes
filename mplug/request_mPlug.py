# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint
import requests
import cv2 as cv

DETECTION_URL = 'http://192.168.196.29:2051/v1/mplug'
IMAGE = 'demo1.jpg'

# Read image as bytes
# with open(IMAGE, 'rb') as f:
#     image_data = f.read()

# Read image as OpenCV array
img = cv.imread(IMAGE)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
image_data = cv.imencode(".jpg", img)[1].tobytes()

response = requests.post(DETECTION_URL, files={'image': image_data}).json()

pprint.pprint(response)

