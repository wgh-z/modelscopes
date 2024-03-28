# funasr==0.8.8
import logging
# import torch
import soundfile

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

model_id = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model_id
)

speech, sample_rate = soundfile.read("zh.wav")  # 我认为跑步最重要的就是给我带来了身体健康
speech_length = speech.shape[0]

sample_offset = 0
chunk_size = [0, 6, 1] #[5, 10, 5] 600ms, [8, 8, 4] 480ms
encoder_chunk_look_back = 3
decoder_chunk_look_back = 1
stride_size =  chunk_size[1] * 960

# is_final = False
# cache={}
# for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
#     if sample_offset + stride_size >= speech_length - 1:
#         stride_size = speech_length - sample_offset
#         is_final = True

#     res = inference_pipeline(speech[sample_offset: sample_offset + stride_size], cache=cache, is_final=is_final, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
#     if len(res):
#         print(res)

cache = {}
total_chunk_num = int(len((speech)-1)/stride_size+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*stride_size:(i+1)*stride_size]
    is_final = i == total_chunk_num - 1
    res = inference_pipeline(
        audio_in=speech_chunk,
        cache=cache,
        is_final=is_final,
        # chunk_size=chunk_size,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back
        )
    if len(res):
        print(res)