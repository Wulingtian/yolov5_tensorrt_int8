# tensorrt模型推理

git clone https://github.com/Wulingtian/yolov5_tensorrt_int8.git（求star）

cd yolov5_tensorrt_int8

vim CMakeLists.txt

修改USER_DIR参数为自己的用户根目录

vim http://yolov5s_infer.cc 修改如下参数

output_name1 output_name2 output_name3 yolov5模型有3个输出

我们可以通过netron查看模型输出名

pip install netron 安装netron

vim netron_yolov5s.py 把如下内容粘贴

    import netron

    netron.start('此处填充简化后的onnx模型路径', port=3344)

python netron_yolov5s.py 即可查看 模型输出名

trt_model_path 量化的的tensorrt推理引擎（models_save目录下trt后缀的文件）

test_img 测试图片路径

INPUT_W INPUT_H 输入图片宽高

NUM_CLASS 训练的模型有多少类

NMS_THRESH nms阈值

CONF_THRESH 置信度

参数配置完毕

mkdir build

cd build

cmake ..

make

./YoloV5sEngine 输出平均推理时间，以及保存预测图片到当前目录下，至此，部署完成！
