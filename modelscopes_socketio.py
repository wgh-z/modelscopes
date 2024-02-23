import base64
from flask import Flask, request  #, render_template, redirect, sessions
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_socketio.namespace import Namespace
# from ultralytics import YOLO
import torch


name_space = '/websocket'
app = Flask(__name__)
CORS(app)
# socketio = SocketIO()
socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*')


class DynamicNamespace(Namespace):
    def __init__(self, namespace):
        super().__init__('/' + namespace)
        self.rtsp = base64.b64decode(namespace).decode('utf-8')
        print(self.rtsp)
        self.connect_num = 0
        self.init_model()

    def init_model(self):
        model = YOLO("yolov8_detection_weight/yolov8m.pt")
        self.results = model(self.rtsp, stream=True)

    def send_result(self):
        while self.connect_num > 0:
            result = next(self.results)
            boxes = result.boxes
            shape = boxes.orig_shape
            boxes = torch.tensor(boxes.data.clone() , dtype=torch.int32)
            boxes = boxes.cpu().numpy().tolist()
            data = {
                'shape': shape,
                'boxes': boxes
            }
            self.emit('message', data)
            socketio.sleep(0.01)
        print('stop send result')

    def on_connect(self):
        self.connect_num += 1
        client_id = request.sid
        print(f'{client_id} connected to dynamic namespace:')
        socketio.start_background_task(target=self.send_result)
        
    def on_disconnect(self):
        self.connect_num = max(0, self.connect_num - 1)
        client_id = request.sid
        print(f'{client_id} disconnected from dynamic namespace:')

    # def on_my_event(self, data):
    #     print('Received data in dynamic namespace:', data)


client_query = []
stopped = False

@socketio.on('connect')
def on_connect():
    namespace_name = request.args.get('rtsp')
    if namespace_name:
        namespace_exists = ('/' + namespace_name) in socketio.server.namespace_handlers
        if not namespace_exists:
            socketio.on_namespace(DynamicNamespace(namespace_name))


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5006, debug=False)
