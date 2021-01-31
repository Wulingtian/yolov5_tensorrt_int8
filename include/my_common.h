#ifndef _MY_COMMON_H_
#define _MY_COMMON_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>

    typedef signed char my_s8;
    typedef unsigned char my_u8;
    typedef signed short my_s16;
    typedef unsigned short my_u16;
    typedef signed int my_s32;
    typedef unsigned int my_u32;
    typedef signed char MY_BOOL;

#define TRUE 1
#define FALSE 0

//#define DEBUG_ON

#ifdef DEBUG_ON
#define MY_DEBUG(...)                                                                          \
    do                                                                                         \
    {                                                                                          \
        fprintf(stdout, "[DEBUG]  %s    %s  (Line  %d) : ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stdout, __VA_ARGS__);                                                          \
    } while (0)
#else
#define MY_DEBUG(...)
#endif

#define MY_ERROR(...)                                                                          \
    do                                                                                         \
    {                                                                                          \
        fprintf(stderr, "[ERROR]  %s    %s  (Line  %d) : ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                                          \
    } while (0)

#define MY_CHECK_NULL(a, errcode)     \
    do                                \
    {                                 \
        if (NULL == (a))              \
        {                             \
            MY_ERROR("NULL DATA \n"); \
            return errcode;           \
        }                             \
    } while (0)

    typedef enum
    {
        MY_SUCCESS = 0, //成功
        MY_FAILED,      //失败
        MY_PARAM_NULL,  //参数为空
        MY_PARAM_SET_ERROR,
        MY_FILE_NOT_EXIST,       //文件不存在
        MY_MEMORY_MALLOC_FAILED, //内存分配失败
        MY_MODEL_LOAD_FAILED,    //模型加载失败
        MY_TENSOR_ALLOC_FAILED,  //tensor内存分配失败
    } result_t;

    typedef enum {
        CPU_MEM_ALLOC = 0,  //Tensor在cpu上
        CPU_MEM_NO_ALLOC,   //上层cpu的数据不会释放，由调用函数保证
        GPU_MEM_ALLOC,      //内部在gpu上分配显存
        GPU_MEM_NO_ALLOC     //由上层分配显存，
    } tensor_memory_type_t;

    typedef struct
    {
        char visibleCard[32];     //设置哪些ＧＰＵ卡是可见的
        int gpu_id;               //虚拟的gpu id
        char model_path[256];     //模型的路径名
        MY_BOOL bIsCipher;            //模型文件是否加密
        int encStartPoint;
        int encLength;
        int maxBatchSize;
        bool bInt8;
        bool bInt16;
    } model_params_t;

    typedef struct
    {
        void *model_handle;         //模型句柄
    } model_handle_t;

    typedef enum
    {
        DT_INVALID = 0,
        DT_FLOAT = 1,
        DT_DOUBLE = 2,
        DT_INT32 = 3,
        DT_UINT8 = 4,
        DT_INT16 = 5,
        DT_INT8 = 6,
        DT_STRING = 7,
        DT_INT64 = 9,
        DT_BOOL = 10,
    } tensor_types_t;

    //Tensor参数的数据结构
    typedef struct
    {
        tensor_types_t type;   //Tensor的类型
        char aTensorName[256]; //Tensor的名字
        int nDims;             //Tensor的rank
        int pShape[8];         //shape
        int nElementSize;      //多少个元素
        int nLength;           //多少个字节长度

        tensor_memory_type_t tensorMemoryType;
        bool bIsOutput;
    } tensor_params_t;

    //定义Tensor的数据结构
    typedef struct
    {
        tensor_params_t *pTensorInfo;
        void *pValue;
    } tensor_t;

    typedef struct
    {
        int nArraySize;
        tensor_params_t *pTensorParamArray;
        char pcSignatureDef[256]; //函数签名
    } tensor_params_array_t;

    typedef struct
    {
        int nArraySize;
        tensor_t *pTensorArray;
        char pcSignatureDef[256]; //函数签名
    } tensor_array_t;

#ifdef __cplusplus
}
#endif

#endif
