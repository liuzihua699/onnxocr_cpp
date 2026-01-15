#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include "types.hpp"

namespace onnxocr {
namespace utils {

// 透视变换裁剪
inline cv::Mat get_rotate_crop_image(const cv::Mat& img, const Box& points) {
    int img_crop_width = static_cast<int>(std::max(
        cv::norm(points[0] - points[1]),
        cv::norm(points[2] - points[3])
    ));
    int img_crop_height = static_cast<int>(std::max(
        cv::norm(points[0] - points[3]),
        cv::norm(points[1] - points[2])
    ));
    
    if (img_crop_width <= 0) img_crop_width = 1;
    if (img_crop_height <= 0) img_crop_height = 1;
    
    std::vector<cv::Point2f> pts_std = {
        {0, 0},
        {static_cast<float>(img_crop_width), 0},
        {static_cast<float>(img_crop_width), static_cast<float>(img_crop_height)},
        {0, static_cast<float>(img_crop_height)}
    };
    
    std::vector<cv::Point2f> src_pts(points.begin(), points.end());
    cv::Mat M = cv::getPerspectiveTransform(src_pts, pts_std);
    
    cv::Mat dst_img;
    cv::warpPerspective(img, dst_img, M, cv::Size(img_crop_width, img_crop_height),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    
    // 如果高宽比大于1.5，旋转90度
    if (dst_img.rows * 1.0f / dst_img.cols >= 1.5f) {
        cv::rotate(dst_img, dst_img, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    return dst_img;
}

// 对检测框进行排序（从上到下，从左到右）
inline std::vector<Box> sorted_boxes(const std::vector<Box>& boxes) {
    std::vector<Box> sorted = boxes;
    
    // 先按y坐标排序，再按x坐标排序
    std::sort(sorted.begin(), sorted.end(), [](const Box& a, const Box& b) {
        if (std::abs(a[0].y - b[0].y) < 10) {
            return a[0].x < b[0].x;
        }
        return a[0].y < b[0].y;
    });
    
    // 冒泡调整同行的框
    for (size_t i = 0; i < sorted.size(); ++i) {
        for (size_t j = i; j > 0; --j) {
            if (std::abs(sorted[j][0].y - sorted[j-1][0].y) < 10 &&
                sorted[j][0].x < sorted[j-1][0].x) {
                std::swap(sorted[j], sorted[j-1]);
            } else {
                break;
            }
        }
    }
    return sorted;
}

// 点顺时针排序
inline Box order_points_clockwise(const Box& pts) {
    Box rect;
    std::vector<float> sums(4), diffs(4);
    for (int i = 0; i < 4; ++i) {
        sums[i] = pts[i].x + pts[i].y;
        diffs[i] = pts[i].y - pts[i].x;
    }
    
    int min_sum_idx = static_cast<int>(std::min_element(sums.begin(), sums.end()) - sums.begin());
    int max_sum_idx = static_cast<int>(std::max_element(sums.begin(), sums.end()) - sums.begin());
    
    rect[0] = pts[min_sum_idx];
    rect[2] = pts[max_sum_idx];
    
    std::vector<int> remaining;
    for (int i = 0; i < 4; ++i) {
        if (i != min_sum_idx && i != max_sum_idx) {
            remaining.push_back(i);
        }
    }
    
    if (diffs[remaining[0]] < diffs[remaining[1]]) {
        rect[1] = pts[remaining[0]];
        rect[3] = pts[remaining[1]];
    } else {
        rect[1] = pts[remaining[1]];
        rect[3] = pts[remaining[0]];
    }
    return rect;
}

// 裁剪检测结果到图像边界内
inline Box clip_box(const Box& box, int img_height, int img_width) {
    Box clipped;
    for (int i = 0; i < 4; ++i) {
        clipped[i].x = std::max(0.0f, std::min(box[i].x, static_cast<float>(img_width - 1)));
        clipped[i].y = std::max(0.0f, std::min(box[i].y, static_cast<float>(img_height - 1)));
    }
    return clipped;
}

} // namespace utils
} // namespace onnxocr
