# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import torch
import time
import cv2 as cv

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_multi_object_tracking.tracker.multitracker import \
    JDETracker
from modelscope.models.cv.video_multi_object_tracking.utils.utils import (
    LoadVideo, cfg_opt)
from modelscope.models.cv.video_single_object_tracking.utils.utils import \
    timestamp_format
from modelscope.models.cv.video_multi_object_tracking.utils.visualization import \
    plot_tracking
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_multi_object_tracking,
    module_name=Pipelines.video_multi_object_tracking)
class VideoMultiObjectTrackingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a multi object tracking pipeline
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        ckpt_path = osp.join(model, ModelFile.TORCH_MODEL_BIN_FILE)
        logger.info(f'loading model from {ckpt_path}')
        opt = cfg_opt()
        self.opt = opt
        self.tracker = JDETracker(opt, ckpt_path, self.device)
        self.tracker.set_buffer_len(kwargs['frame_rate'])
        logger.info('init tracker done')

    def preprocess(self, input) -> Input:
        self.video_path = input[0]
        return input

    def forward(self, img) -> Dict[str, Any]:
        output_boxes = []
        output_labels = []

        # blob = torch.from_numpy(img).unsqueeze(0)
        blob = img.unsqueeze(0)  # add batch dimension
        online_targets = self.tracker.update(blob)
        # online_tlwhs = []
        # online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # vertical = tlwh[2] / tlwh[3] > 1.6
            # if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
            #     online_tlwhs.append([
            #         tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
            #     ])
            #     online_ids.append(tid)
            # output_boxes.append([  # [x1, y1, x2, y2]
            #     int(max(0, tlwh[0])),
            #     int(max(0, tlwh[1])),
            #     int(tlwh[0] + tlwh[2]),
            #     int(tlwh[1] + tlwh[3])
            # ])
            output_boxes.append([  # [x1, y1, x2, y2]
                int(max(0, tlwh[0])),
                int(max(0, tlwh[1])),
                int(tlwh[2]),
                int(tlwh[3])
            ])
            output_labels.append(tid)

        return {
            OutputKeys.BOXES: output_boxes,
            OutputKeys.LABELS: output_labels,
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
    

def run(model_id, video_path):
    # dataloader = LoadVideo(video_path, img_size=(640, 640))
    dataloader = LoadVideo(video_path)

    size = (1920, 1080)
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv.VideoWriter('out.avi', fourcc,
                                    dataloader.frame_rate, size,
                                    True)

    tracker = VideoMultiObjectTrackingPipeline(model_id,
                                               device='gpu',
                                               frame_rate=dataloader.frame_rate,
                                               )
    avg_fps = 0
    for i, (path, img, img0) in enumerate(dataloader):
        start_time = time.time()
        result = tracker(img)
        end_time = time.time()
        avg_fps = (avg_fps + (1 / (end_time - start_time))) / 2
        # print(result)
        print(f'frame {i} done, fps: {avg_fps}')
        frame = plot_tracking(img0, result[OutputKeys.BOXES], result[OutputKeys.LABELS], fps=avg_fps)
        video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':
    model_id = './weight/damo/cv_yolov5_video-multi-object-tracking_fairmot'
    video_path = r'1708416037279.mp4'

    run(model_id, video_path)
