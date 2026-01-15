/**
 * OnnxOCR C API - 动态库对外接口
 */

#ifndef ONNXOCR_API_H
#define ONNXOCR_API_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #ifdef ONNXOCR_EXPORTS
        #define ONNXOCR_API __declspec(dllexport)
    #else
        #define ONNXOCR_API __declspec(dllimport)
    #endif
#else
    #define ONNXOCR_API __attribute__((visibility("default")))
#endif

// OCR句柄
typedef void* OcrHandle;

// 单个文本行结果
typedef struct {
    float box[8];      // 四个点坐标 [x0,y0,x1,y1,x2,y2,x3,y3]
    char* text;        // 识别文本 (UTF-8)
    float score;       // 置信度
} OcrTextLine;

// OCR结果
typedef struct {
    OcrTextLine* lines;  // 文本行数组
    int count;           // 文本行数量
} OcrResult;

// 配置结构
typedef struct {
    const char* det_model_path;      // 检测模型路径
    const char* rec_model_path;      // 识别模型路径
    const char* cls_model_path;      // 分类模型路径
    const char* rec_char_dict_path;  // 字典文件路径
    int use_gpu;                     // 是否使用GPU (0=CPU, 1=GPU)
    int gpu_id;                      // GPU设备ID
    int use_angle_cls;               // 是否使用方向分类
    float drop_score;                // 过滤阈值
} OcrConfig;

/**
 * 创建OCR实例
 * @param config 配置
 * @return OCR句柄，失败返回NULL
 */
ONNXOCR_API OcrHandle ocr_create(const OcrConfig* config);

/**
 * 销毁OCR实例
 * @param handle OCR句柄
 */
ONNXOCR_API void ocr_destroy(OcrHandle handle);

/**
 * 识别图片文件
 * @param handle OCR句柄
 * @param image_path 图片路径
 * @param result 输出结果
 * @return 0成功，非0失败
 */
ONNXOCR_API int ocr_recognize_file(OcrHandle handle, const char* image_path, OcrResult* result);

/**
 * 识别内存中的图片数据
 * @param handle OCR句柄
 * @param image_data 图片数据 (BGR格式)
 * @param width 图片宽度
 * @param height 图片高度
 * @param channels 通道数 (3)
 * @param result 输出结果
 * @return 0成功，非0失败
 */
ONNXOCR_API int ocr_recognize_buffer(OcrHandle handle, const unsigned char* image_data, 
                                      int width, int height, int channels, OcrResult* result);

/**
 * 释放OCR结果
 * @param result 结果指针
 */
ONNXOCR_API void ocr_free_result(OcrResult* result);

/**
 * 获取最后一次错误信息
 * @return 错误信息字符串
 */
ONNXOCR_API const char* ocr_get_last_error();

#ifdef __cplusplus
}
#endif

#endif // ONNXOCR_API_H
