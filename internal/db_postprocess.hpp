#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include "types.hpp"
#include "clipper.hpp"  // from third_party/

namespace onnxocr {
namespace postprocess {

class DBPostProcess {
public:
    DBPostProcess(float thresh = 0.3f, float box_thresh = 0.7f,
                  int max_candidates = 1000, float unclip_ratio = 2.0f,
                  bool use_dilation = false, const std::string& score_mode = "fast",
                  const std::string& box_type = "quad")
        : thresh_(thresh), box_thresh_(box_thresh), max_candidates_(max_candidates),
          unclip_ratio_(unclip_ratio), use_dilation_(use_dilation),
          score_mode_(score_mode), box_type_(box_type), min_size_(3) {}
    
    std::vector<Box> operator()(const float* pred, int height, int width,
                                 int src_h, int src_w) {
        // 创建预测图
        cv::Mat pred_mat(height, width, CV_32FC1);
        for (int i = 0; i < height; ++i) {
            float* row = pred_mat.ptr<float>(i);
            for (int j = 0; j < width; ++j) {
                row[j] = pred[i * width + j];
            }
        }
        
        // 二值化
        cv::Mat bitmap(height, width, CV_8UC1);
        for (int i = 0; i < height; ++i) {
            uchar* dst_row = bitmap.ptr<uchar>(i);
            const float* src_row = pred_mat.ptr<float>(i);
            for (int j = 0; j < width; ++j) {
                dst_row[j] = src_row[j] > thresh_ ? 255 : 0;
            }
        }
        
        // 膨胀
        if (use_dilation_) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::dilate(bitmap, bitmap, kernel);
        }
        
        // 提取轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
        std::vector<Box> boxes;
        int num_contours = std::min(static_cast<int>(contours.size()), max_candidates_);
        
        for (int i = 0; i < num_contours; ++i) {
            auto& contour = contours[i];
            
            if (contour.size() < 4) continue;
            
            // 获取最小外接矩形
            std::vector<cv::Point2f> pts;
            float sside;
            std::tie(pts, sside) = get_mini_boxes(contour);
            
            if (sside < min_size_) continue;
            
            // 计算分数
            float score = box_score_fast(pred_mat, pts);
            if (score < box_thresh_) continue;
            
            // Unclip
            auto expanded = unclip(pts, unclip_ratio_);
            if (expanded.empty()) continue;
            
            // 再次获取最小外接矩形
            std::vector<cv::Point2f> box;
            float sside2;
            std::tie(box, sside2) = get_mini_boxes(expanded);
            
            if (sside2 < min_size_ + 2) continue;
            
            // 映射到原图坐标
            Box result_box;
            for (int j = 0; j < 4; ++j) {
                float x = box[j].x / width * src_w;
                float y = box[j].y / height * src_h;
                result_box[j].x = std::max(0.0f, std::min(x, static_cast<float>(src_w)));
                result_box[j].y = std::max(0.0f, std::min(y, static_cast<float>(src_h)));
            }
            boxes.push_back(result_box);
        }
        
        return boxes;
    }

private:
    std::pair<std::vector<cv::Point2f>, float> get_mini_boxes(
        const std::vector<cv::Point>& contour) {
        cv::RotatedRect bounding_box = cv::minAreaRect(contour);
        cv::Point2f pts[4];
        bounding_box.points(pts);
        
        std::vector<cv::Point2f> points(pts, pts + 4);
        std::sort(points.begin(), points.end(), 
                  [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
        
        std::vector<cv::Point2f> box(4);
        if (points[1].y > points[0].y) {
            box[0] = points[0];
            box[3] = points[1];
        } else {
            box[0] = points[1];
            box[3] = points[0];
        }
        if (points[3].y > points[2].y) {
            box[1] = points[2];
            box[2] = points[3];
        } else {
            box[1] = points[3];
            box[2] = points[2];
        }
        
        return {box, std::min(bounding_box.size.width, bounding_box.size.height)};
    }
    
    std::pair<std::vector<cv::Point2f>, float> get_mini_boxes(
        const std::vector<cv::Point2f>& contour) {
        std::vector<cv::Point> int_contour;
        for (const auto& pt : contour) {
            int_contour.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
        }
        return get_mini_boxes(int_contour);
    }
    
    float box_score_fast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box) {
        int h = bitmap.rows, w = bitmap.cols;
        
        float xmin_f = std::min({box[0].x, box[1].x, box[2].x, box[3].x});
        float xmax_f = std::max({box[0].x, box[1].x, box[2].x, box[3].x});
        float ymin_f = std::min({box[0].y, box[1].y, box[2].y, box[3].y});
        float ymax_f = std::max({box[0].y, box[1].y, box[2].y, box[3].y});
        
        int xmin = std::max(0, static_cast<int>(std::floor(xmin_f)));
        int xmax = std::min(w - 1, static_cast<int>(std::ceil(xmax_f)));
        int ymin = std::max(0, static_cast<int>(std::floor(ymin_f)));
        int ymax = std::min(h - 1, static_cast<int>(std::ceil(ymax_f)));
        
        if (xmax <= xmin || ymax <= ymin) return 0.0f;
        
        cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
        std::vector<cv::Point> pts(4);
        for (int i = 0; i < 4; ++i) {
            pts[i] = cv::Point(static_cast<int>(box[i].x - xmin), 
                               static_cast<int>(box[i].y - ymin));
        }
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(1));
        
        cv::Mat roi = bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
        return static_cast<float>(cv::mean(roi, mask)[0]);
    }
    
    std::vector<cv::Point2f> unclip(const std::vector<cv::Point2f>& box, float unclip_ratio) {
        // 计算多边形面积和周长
        double area = cv::contourArea(box);
        double length = cv::arcLength(box, true);
        
        if (length < 1e-6) return {};
        
        double distance = area * unclip_ratio / length;
        
        // 使用Clipper库进行扩展
        ClipperLib::Path subj;
        for (const auto& pt : box) {
            subj << ClipperLib::IntPoint(
                static_cast<ClipperLib::cInt>(pt.x * 1000), 
                static_cast<ClipperLib::cInt>(pt.y * 1000));
        }
        
        ClipperLib::Paths solution;
        ClipperLib::ClipperOffset co;
        co.AddPath(subj, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
        co.Execute(solution, distance * 1000);
        
        if (solution.empty() || solution[0].empty()) return {};
        
        std::vector<cv::Point2f> result;
        for (const auto& pt : solution[0]) {
            result.push_back(cv::Point2f(
                static_cast<float>(pt.X / 1000.0), 
                static_cast<float>(pt.Y / 1000.0)));
        }
        return result;
    }

    float thresh_;
    float box_thresh_;
    int max_candidates_;
    float unclip_ratio_;
    bool use_dilation_;
    std::string score_mode_;
    std::string box_type_;
    int min_size_;
};

} // namespace postprocess
} // namespace onnxocr
