convert tf model to onnx model
+ stride: 16
+ supported image size: 129x129, 257x257, 513x513

onnx model name: mobilenet_stride16.onnx

update the variables in source code: tget_wid, tget_hei with above image size

run below commands: 

$make

$./poseNet_camera -model mobilenet -stride 16