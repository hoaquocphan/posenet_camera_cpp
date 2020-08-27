
onnxruntime: PoseNet_camera.cpp
	${CXX} -std=c++14 PoseNet_camera.cpp \
	-DONNX_ML \
	-I ${PWD} \
	-L ${SDKTARGETSYSROOT}/usr/lib64/ \
	-L ${SDKTARGETSYSROOT}/usr/lib64/onnx/ \
	-L ${SDKTARGETSYSROOT}/usr/lib64/external/protobuf/cmake/ \
	-L ${SDKTARGETSYSROOT}/usr/lib64/external/re2/ \
	-lonnxruntime_session \
	-lonnxruntime_providers \
	-lautoml_featurizers \
	-lonnxruntime_framework \
	-lonnxruntime_optimizer \
	-lonnxruntime_graph \
	-lonnxruntime_common \
	-lonnx_proto \
	-lprotobuf \
	-lre2 \
	-lonnxruntime_util \
	-lonnxruntime_mlas \
	-lonnx \
	-ljpeg -ltbb -ltiff -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_photo -lopencv_imgcodecs \
	-lpthread -O2 -fopenmp -ldl ${LDFLAGS} -o poseNet_camera

clean:
	rm -rf *.o poseNet_camera
