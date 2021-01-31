cmake_minimum_required(VERSION 3.15)
project(YoloV5sEngine)

set(CMAKE_CXX_STANDARD 14)

SET(USER_DIR /home/willer)

SET(CUDA_INC_DIR /usr/local/cuda-11.0/include)
SET(CUDNN_INC_DIR /usr/local/cuda-11.0/include)
SET(TENSORRT_INC_DIR ${USER_DIR}/TensorRT-7.2.1.6/include)
SET(TRT_ENGINE_INC_DIR include)
SET(INC_DIR ${CUDA_INC_DIR} ${CUDNN_INC_DIR} ${TRT_ENGINE_INC_DIR} ${TRT_ENGINE_INC_DIR})

SET(CUDA_LIB_DIR /usr/local/cuda-11.0/lib64)
SET(CUDNN_LIB_DIR /usr/local/cuda-11.0/lib64)
SET(TENSORRT_LIB_DIR ${USER_DIR}/TensorRT-7.2.1.6/lib)
SET(TRT_ENGINE_LIB_DIR trt_engine_lib/)
SET(LIB_DIR ${CUDA_LIB_DIR} ${CUDNN_LIB_DIR} ${TENSORRT_LIB_DIR} ${TRT_ENGINE_LIB_DIR})

set(TRT_LIBS nvinfer nvinfer_plugin )
set(CUDNN_LIBs cudnn )
set(TRT_ENGINE_LIBS MyTtrEngine-trt721)
set(CUDA_LIBS cublas cudart curand cufft nppig)

set(LINK_LIBS ${TRT_LIBS} ${CUDNN_LIBs} ${TRT_ENGINE_LIBS}  ${CUDA_LIBS})

set(OpenCV_INCLUDE_DIRS ${USER_DIR}/opencv-3.4.2/include)
set(OpenCV_LIBS_DIR ${USER_DIR}/opencv-3.4.2/build/lib)
set(OpenCV_LIBS opencv_core opencv_dnn opencv_imgproc opencv_imgcodecs opencv_highgui)

include_directories(${INC_DIR} ${OpenCV_INCLUDE_DIRS})
link_directories(${LIB_DIR} ${OpenCV_LIBS_DIR})
link_libraries(${LINK_LIBS} ${OpenCV_LIBS})

add_executable(YoloV5sEngine
        yolov5s_infer.cc)
