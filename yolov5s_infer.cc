
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include  "my_interface.h"
#include <chrono>
#include <vector>
#include <npp.h>

#define INPUT_W 640
#define INPUT_H 640
#define IsPadding 1
#define NUM_CLASS 1
#define NMS_THRESH 0.3
#define CONF_THRESH 0.6
char* output_name1 = "output";
char* output_name2 = "317";
char* output_name3 = "337";
char* trt_model_path = "../models/yolov5s-4.0-int8-relu.trt";
std::string test_img = "../test_imgs/2bb75da9-331a-3d4f-96d0-5817ae6aed80.jpg";

using namespace cv;
using namespace std;

struct Bbox{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int cid;
};

struct Anchor{
    float width;
    float height;
};

std::vector<Anchor> initAnchors(){
    std::vector<Anchor> anchors;
    Anchor anchor;
    // 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326
    anchor.width = 10;
    anchor.height = 13;
    anchors.emplace_back(anchor);
    anchor.width = 16;
    anchor.height = 30;
    anchors.emplace_back(anchor);
    anchor.width = 32;
    anchor.height = 23;
    anchors.emplace_back(anchor);
    anchor.width = 30;
    anchor.height = 61;
    anchors.emplace_back(anchor);
    anchor.width = 62;
    anchor.height = 45;
    anchors.emplace_back(anchor);
    anchor.width = 59;
    anchor.height = 119;
    anchors.emplace_back(anchor);
    anchor.width = 116;
    anchor.height = 90;
    anchors.emplace_back(anchor);
    anchor.width = 156;
    anchor.height = 198;
    anchors.emplace_back(anchor);
    anchor.width = 373;
    anchor.height = 326;
    anchors.emplace_back(anchor);
    return anchors;
}

template <typename T>
T clip(const T &n, const T &lower, const T &upper){
    return std::max(lower, std::min(n, upper));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

void transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<Bbox> &bboxes,
               bool is_padding) {
    if(is_padding){
        float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw), static_cast<float>(oh) / static_cast<float>(ih));
        int nh = static_cast<int>(scale * static_cast<float>(ih));
        int nw = static_cast<int>(scale * static_cast<float>(iw));
        int dh = (oh - nh) / 2;
        int dw = (ow - nw) / 2;
        for (auto &bbox : bboxes){
            bbox.xmin = (bbox.xmin - dw) / scale;
            bbox.ymin = (bbox.ymin - dh) / scale;
            bbox.xmax = (bbox.xmax - dw) / scale;
            bbox.ymax = (bbox.ymax - dh) / scale;
        }
    }else{
        for (auto &bbox : bboxes){
            bbox.xmin = bbox.xmin * iw / ow;
            bbox.ymin = bbox.ymin * ih / oh;
            bbox.xmax = bbox.xmax * iw / ow;
            bbox.ymax = bbox.ymax * ih / oh;
        }
    }
}


cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes){
    for (auto it: bboxes){
        float score = it.score;
        cv::rectangle(image, cv::Point(it.xmin, it.ymin), cv::Point(it.xmax, it.ymax), cv::Scalar(255, 204,0), 3);
        cv::putText(image, std::to_string(score), cv::Point(it.xmin, it.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }
    return image;
}

void nms_cpu(std::vector<Bbox> &bboxes, float threshold) {
    if (bboxes.empty()){
        return ;
    }
    // 1.之前需要按照score排序
    std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score>b2.score;});
    // 2.先求出所有bbox自己的大小
    std::vector<float> area(bboxes.size());
    for (int i=0; i<bboxes.size(); ++i){
        area[i] = (bboxes[i].xmax - bboxes[i].xmin + 1) * (bboxes[i].ymax - bboxes[i].ymin + 1);
    }
    // 3.循环
    for (int i=0; i<bboxes.size(); ++i){
        for (int j=i+1; j<bboxes.size(); ){
            float left = std::max(bboxes[i].xmin, bboxes[j].xmin);
            float right = std::min(bboxes[i].xmax, bboxes[j].xmax);
            float top = std::max(bboxes[i].ymin, bboxes[j].ymin);
            float bottom = std::min(bboxes[i].ymax, bboxes[j].ymax);
            float width = std::max(right - left + 1, 0.f);
            float height = std::max(bottom - top + 1, 0.f);
            float u_area = height * width;
            float iou = (u_area) / (area[i] + area[j] - u_area);
            if (iou>=threshold){
                bboxes.erase(bboxes.begin()+j);
                area.erase(area.begin()+j);
            }else{
                ++j;
            }
        }
    }
}
template <typename T>
T sigmoid(const T &n) {
    return 1 / (1 + exp(-n));
}
void postProcessParall(const int height, const int width, int scale_idx, float postThres, tensor_t * origin_output, vector<int> Strides, vector<Anchor> Anchors, vector<Bbox> *bboxes)
{
    Bbox bbox;
    float cx, cy, w_b, h_b, score;
    int cid;
    const float *ptr = (float *)origin_output->pValue;
    for(unsigned long a=0; a<3; ++a){
        for(unsigned long h=0; h<height; ++h){
            for(unsigned long w=0; w<width; ++w){
                const float *cls_ptr =  ptr + 5;
                cid = argmax(cls_ptr, cls_ptr+NUM_CLASS);
                score = sigmoid(ptr[4]) * sigmoid(cls_ptr[cid]);
                if(score>=postThres){
                    cx = (sigmoid(ptr[0]) * 2.f - 0.5f + static_cast<float>(w)) * static_cast<float>(Strides[scale_idx]);
                    cy = (sigmoid(ptr[1]) * 2.f - 0.5f + static_cast<float>(h)) * static_cast<float>(Strides[scale_idx]);
                    w_b = powf(sigmoid(ptr[2]) * 2.f, 2) * Anchors[scale_idx * 3 + a].width;
                    h_b = powf(sigmoid(ptr[3]) * 2.f, 2) * Anchors[scale_idx * 3 + a].height;
                    bbox.xmin = clip(cx - w_b / 2, 0.F, static_cast<float>(INPUT_W - 1));
                    bbox.ymin = clip(cy - h_b / 2, 0.f, static_cast<float>(INPUT_H - 1));
                    bbox.xmax = clip(cx + w_b / 2, 0.f, static_cast<float>(INPUT_W - 1));
                    bbox.ymax = clip(cy + h_b / 2, 0.f, static_cast<float>(INPUT_H - 1));
                    bbox.score = score;
                    bbox.cid = cid;
                    //std::cout<< "bbox.cid : " << bbox.cid << std::endl;
                    bboxes->push_back(bbox);
                }
                ptr += 5 + NUM_CLASS;
            }
        }
    }
}
vector<Bbox> postProcess(vector<tensor_t *> origin_output, float postThres, float nmsThres) {

    vector<Anchor> Anchors = initAnchors();
    vector<Bbox> bboxes;
    vector<int> Strides = vector<int> {8, 16, 32};
    for (int scale_idx=0; scale_idx<3; ++scale_idx) {
        const int stride = Strides[scale_idx];
        const int width = (INPUT_W + stride - 1) / stride;
        const int height = (INPUT_H + stride - 1) / stride;
        //std::cout << "width : " << width << " " << "height : " << height << std::endl;
        tensor_t * cur_output_tensor = origin_output[scale_idx];
        postProcessParall(height, width, scale_idx, postThres, cur_output_tensor, Strides, Anchors, &bboxes);
    }
    nms_cpu(bboxes, nmsThres);
    return bboxes;
}

void cudaResize(cv::Mat &image, cv::Mat &rsz_img)
{
    int outsize = rsz_img.cols * rsz_img.rows * sizeof(uchar3);

    int inwidth = image.cols;
    int inheight = image.rows;
    int memSize = inwidth * inheight * sizeof(uchar3);

    NppiSize srcsize = {inwidth, inheight};
    NppiRect srcroi  = {0, 0, inwidth, inheight};
    NppiSize dstsize = {rsz_img.cols, rsz_img.rows};
    NppiRect dstroi  = {0, 0, rsz_img.cols, rsz_img.rows};

    uchar3* d_src = NULL;
    uchar3* d_dst = NULL;
    cudaMalloc((void**)&d_src, memSize);
    cudaMalloc((void**)&d_dst, outsize);
    cudaMemcpy(d_src, image.data, memSize, cudaMemcpyHostToDevice);

    // nvidia npp 图像处理
    nppiResize_8u_C3R( (Npp8u*)d_src, inwidth * 3, srcsize, srcroi,
                       (Npp8u*)d_dst, rsz_img.cols * 3, dstsize, dstroi,
                       NPPI_INTER_LINEAR );


    cudaMemcpy(rsz_img.data, d_dst, outsize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}
cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    //cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cudaResize(img,re);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

/////
int main(int argc, const char *argv[]) {
    float total = 0, ms, pr_ms, po_ms;
    int test_echo = 20;

    // 创建输入输出tensor结构体
    tensor_params_array_t in_tensor_params_ar = {0};
    tensor_params_array_t out_tensor_params_ar = {0};
    tensor_array_t *input_tensor_array = NULL;
    tensor_array_t *ouput_tensor_array = NULL;

    /****************** */
    // 定义输入tensor
    in_tensor_params_ar.nArraySize = 1;
    in_tensor_params_ar.pTensorParamArray = (tensor_params_t *) malloc(
            in_tensor_params_ar.nArraySize * sizeof(tensor_params_t));
    memset(in_tensor_params_ar.pTensorParamArray, 0, in_tensor_params_ar.nArraySize * sizeof(tensor_params_t));

    tensor_params_t *cur_in_tensor_params = in_tensor_params_ar.pTensorParamArray;

    // 第一个输入tensor
    cur_in_tensor_params[0].nDims = 4;
    cur_in_tensor_params[0].type = DT_FLOAT;
    cur_in_tensor_params[0].pShape[0] = 1; //batch size can't set to -1
    cur_in_tensor_params[0].pShape[1] = 3;
    cur_in_tensor_params[0].pShape[2] = INPUT_H;
    cur_in_tensor_params[0].pShape[3] = INPUT_W;
    strcpy(cur_in_tensor_params[0].aTensorName, "images");
    cur_in_tensor_params[0].tensorMemoryType = CPU_MEM_ALLOC;

    /*************** */
    // 定义输出tensor
    out_tensor_params_ar.nArraySize = 3;
    out_tensor_params_ar.pTensorParamArray = (tensor_params_t *) malloc(
            out_tensor_params_ar.nArraySize * sizeof(tensor_params_t));
    memset(out_tensor_params_ar.pTensorParamArray, 0, out_tensor_params_ar.nArraySize * sizeof(tensor_params_t));

    tensor_params_t *cur_out_tensor_params = out_tensor_params_ar.pTensorParamArray;

    cur_out_tensor_params[0].nDims = 5;
    cur_out_tensor_params[0].type = DT_FLOAT;
    cur_out_tensor_params[0].pShape[0] = 1;
    cur_out_tensor_params[0].pShape[1] = 3;
    cur_out_tensor_params[0].pShape[2] = INPUT_H/8;
    cur_out_tensor_params[0].pShape[3] = INPUT_H/8;
    cur_out_tensor_params[0].pShape[4] = NUM_CLASS+5;
    cur_out_tensor_params[0].tensorMemoryType = CPU_MEM_ALLOC;
    strcpy(cur_out_tensor_params[0].aTensorName, output_name1);

    cur_out_tensor_params[1].nDims = 5;
    cur_out_tensor_params[1].type = DT_FLOAT;
    cur_out_tensor_params[1].pShape[0] = 1;
    cur_out_tensor_params[1].pShape[1] = 3;
    cur_out_tensor_params[1].pShape[2] = INPUT_H/16;
    cur_out_tensor_params[1].pShape[3] = INPUT_H/16;
    cur_out_tensor_params[1].pShape[4] = NUM_CLASS+5;
    cur_out_tensor_params[1].tensorMemoryType = CPU_MEM_ALLOC;
    strcpy(cur_out_tensor_params[1].aTensorName, output_name2);

    cur_out_tensor_params[2].nDims = 5;
    cur_out_tensor_params[2].type = DT_FLOAT;
    cur_out_tensor_params[2].pShape[0] = 1;
    cur_out_tensor_params[2].pShape[1] = 3;
    cur_out_tensor_params[2].pShape[2] = INPUT_H/32;
    cur_out_tensor_params[2].pShape[3] = INPUT_H/32;
    cur_out_tensor_params[2].pShape[4] = NUM_CLASS+5;
    cur_out_tensor_params[2].tensorMemoryType = CPU_MEM_ALLOC;
    strcpy(cur_out_tensor_params[2].aTensorName, output_name3);


    // 初始化输入输出结构体，分配内存
    if (my_init_tensors(&in_tensor_params_ar, &out_tensor_params_ar,
                        &input_tensor_array, &ouput_tensor_array) != MY_SUCCESS) {
        printf("Open Internal memory error!\n");
    }

    //===================obtain Handle=========================================
    model_params_t tModelParam = {0}; //model input parameter
    model_handle_t tModelHandle = {0};

    strcpy(tModelParam.visibleCard, "0");
    tModelParam.gpu_id = 0; //GPU 0
    tModelParam.bIsCipher = FALSE;
    tModelParam.maxBatchSize = 1;
//    strcpy(tModelParam.model_path, "../models/TRT_ssd_mobilenet_v2_coco.trt");

    //strcpy(tModelParam.model_path, "../models/yolov5s-simple-2.0.trt");
    strcpy(tModelParam.model_path, trt_model_path);
//    strcpy(tModelParam.model_path, "../models/");

//  tModelParam.bIsCipher = TRUE;
//  tModelParam.encStartPoint = 340;
//  tModelParam.encLength = 5000;
//  strcpy(tModelParam.model_path, "models/encrpy_model");

    //call API open model
    if (my_load_model(&tModelParam,
                      input_tensor_array,
                      ouput_tensor_array,
                      &tModelHandle) != MY_SUCCESS) {
        printf("Open model error!\n");
    }
    std::cout << "Load model sucess\n";


    //string file_name = "/home/willer/yolov5-3.1/data/coco/images/val2017/21.jpg";
    string file_name = test_img;
    tensor_t *cur_input_tensor_image = &(input_tensor_array->pTensorArray[0]);

    cv::Mat cImage;
    cImage = cv::imread(file_name);
    std::cout << "Read img finished!\n";
    cv::Mat showImage = cImage.clone();


    static float data[3 * INPUT_H * INPUT_W];

    auto pr_start = std::chrono::high_resolution_clock::now();
    cv::Mat pre_img = preprocess_img(cImage);
    //auto pr_end = std::chrono::high_resolution_clock::now();
    //pr_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
    std::cout << "preprocess_img finished!\n";
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pre_img.data + row * pre_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    auto pr_end = std::chrono::high_resolution_clock::now();
    pr_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();

    memcpy((float *) (cur_input_tensor_image->pValue),
           data, 3 * INPUT_H * INPUT_W * sizeof(float));


    printf("----->memcpy data is success......\n");
    for (int j = 0; j < test_echo; ++j) {
        auto t_start = std::chrono::high_resolution_clock::now();

        my_inference_tensors(&tModelHandle);

        auto t_end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        total += ms;
        std::cout << "[ " << j << " ] " << ms << " ms." << std::endl;
    }

    total /= test_echo;
    std::cout << "Average over " << test_echo << " runs is " << total << " ms." << std::endl;

    tensor_t *cur_output_tensor_0 = &(ouput_tensor_array->pTensorArray[0]);
    tensor_t *cur_output_tensor_1 = &(ouput_tensor_array->pTensorArray[1]);
    tensor_t *cur_output_tensor_2 = &(ouput_tensor_array->pTensorArray[2]);
    vector<tensor_t *> cur_output_tensors;
    cur_output_tensors.push_back(cur_output_tensor_0);
    cur_output_tensors.push_back(cur_output_tensor_1);
    cur_output_tensors.push_back(cur_output_tensor_2);

    auto po_start = std::chrono::high_resolution_clock::now();
    vector<Bbox> bboxes = postProcess(cur_output_tensors, CONF_THRESH, NMS_THRESH);
    auto po_end = std::chrono::high_resolution_clock::now();
    po_ms = std::chrono::duration<float, std::milli>(po_end - po_start).count();

    transform(showImage.rows, showImage.cols, INPUT_W, INPUT_H, bboxes, IsPadding);

    showImage = renderBoundingBox(showImage, bboxes);
    cv::imwrite("final.jpg", showImage);

    //std::cout << "prepareImage " << " runs is " << pr_ms << " ms." << std::endl;
    //std::cout << "postProcess " << " runs is " << po_ms << " ms." << std::endl;


    my_deinit_tensors(input_tensor_array, ouput_tensor_array);

    my_release_model(&tModelHandle);

    std::cout << "complete!!!" << std::endl;

    return 0;
}
