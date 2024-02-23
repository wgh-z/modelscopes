from modelscope.models import Model
from modelscope.exporters import Exporter
model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'
model = Model.from_pretrained(model_id)
output_files = Exporter.from_model(model).export_onnx(opset=13, output_dir='/tmp', ...)
print(output_files)