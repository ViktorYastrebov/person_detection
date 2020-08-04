* Common. It's the demo to show how to use TensorRT inference engine.

* Models:
  Use all pretrained models from the pytorch_demo folder.
  Convert them to the onnx format
  then install:
	> pip install onnxsimplifier
  run python:
	> python3 -m onnxsim yolov3-spp.onnx yolov3-spp.simplified.onnx
	
	