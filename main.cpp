/**
 * Deepfake Detection - Optimized C++ Version
 *
 * Usage: deepfake_detector --image <path> [options]
 *
 * Options:
 *   --image <path>       Input image path (required)
 *   --detector <path>    Face detector model (default: det_10g.onnx)
 *   --gend <path>        GenD model (default: gend.onnx)
 *   --output <path>      Output image path (default: result.jpg)
 *   --thresh <float>     Detection threshold (default: 0.5)
 *   --scale <float>      Face alignment scale (default: 1.3)
 *   --max-faces <int>    Max faces to detect (default: unlimited)
 *   --gpu                Use GPU (CUDA)
 *   --help               Show help
 */

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <Windows.h>

std::wstring stringToWstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
    return wstr;
}
#endif

namespace deepfake {

    // ============================================================================
    // Data Structures
    // ============================================================================

    struct FaceLandmarks {
        cv::Point2f points[5];
    };

    struct FaceDetection {
        cv::Rect2f bbox;
        float confidence;
        FaceLandmarks landmarks;
    };

    struct FaceResult {
        cv::Rect2f bbox;
        float fake_prob;
    };

    struct ImageResult {
        std::vector<FaceResult> faces;
        float avg_fake_prob;
        std::string image_path;
    };

    // ============================================================================
    // Face Alignment
    // ============================================================================

    int computeTargetSize(const FaceLandmarks& landmarks, float scale) {
        constexpr float dst[5][2] = {
            {0.34f, 0.46f}, {0.66f, 0.46f}, {0.50f, 0.64f}, {0.37f, 0.82f}, {0.63f, 0.82f}
        };

        std::vector<float> desired_dists, dst_dists;
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                float dx = landmarks.points[i].x - landmarks.points[j].x;
                float dy = landmarks.points[i].y - landmarks.points[j].y;
                desired_dists.push_back(std::sqrt(dx * dx + dy * dy));

                float ddx = dst[i][0] - dst[j][0];
                float ddy = dst[i][1] - dst[j][1];
                dst_dists.push_back(std::sqrt(ddx * ddx + ddy * ddy));
            }
        }

        float sum_ratio = 0.0f;
        for (size_t i = 0; i < desired_dists.size(); i++) {
            sum_ratio += desired_dists[i] / dst_dists[i];
        }
        return static_cast<int>(std::round((sum_ratio / desired_dists.size()) * scale));
    }

    cv::Mat alignFace(const cv::Mat& img, const FaceLandmarks& landmarks, float scale = 1.3f) {
        constexpr float dst[5][2] = {
            {0.34f, 0.46f}, {0.66f, 0.46f}, {0.50f, 0.64f}, {0.37f, 0.82f}, {0.63f, 0.82f}
        };

        int target_size = computeTargetSize(landmarks, scale);
        int width = target_size, height = target_size;

        std::vector<cv::Point2f> src_pts, dst_pts;
        for (int i = 0; i < 5; i++) {
            src_pts.push_back(landmarks.points[i]);
            dst_pts.push_back(cv::Point2f(dst[i][0] * width, dst[i][1] * height));
        }

        float margin_rate = scale - 1.0f;
        float x_margin = width * margin_rate / 2.0f;
        float y_margin = height * margin_rate / 2.0f;

        for (int i = 0; i < 5; i++) {
            dst_pts[i].x = (dst_pts[i].x + x_margin) * width / (width + 2 * x_margin);
            dst_pts[i].y = (dst_pts[i].y + y_margin) * height / (height + 2 * y_margin);
        }

        cv::Mat M = cv::estimateAffinePartial2D(src_pts, dst_pts, cv::noArray(), cv::LMEDS);
        if (M.empty()) {
            M = cv::Mat::zeros(2, 3, CV_64F);
            M.at<double>(0, 0) = 1.0;
            M.at<double>(1, 1) = 1.0;
        }

        cv::Mat aligned;
        cv::warpAffine(img, aligned, M, cv::Size(width, height), cv::INTER_LINEAR);
        return aligned;
    }

    // ============================================================================
    // Image Preprocessor
    // ============================================================================

    class Preprocessor {
    public:
        std::vector<float> process(const cv::Mat& image_rgb, int size = 224) {
            cv::Mat resized;
            cv::resize(image_rgb, resized, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);

            cv::Mat float_img;
            resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

            constexpr float mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
            constexpr float std[3] = { 0.26862954f, 0.26130258f, 0.27577711f };

            std::vector<float> output(3 * size * size);
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < size; h++) {
                    for (int w = 0; w < size; w++) {
                        float val = float_img.at<cv::Vec3f>(h, w)[c];
                        output[c * size * size + h * size + w] = (val - mean[c]) / std[c];
                    }
                }
            }
            return output;
        }

        std::vector<int64_t> getInputShape(int size = 224) {
            return { 1, 3, size, size };
        }
    };

    // ============================================================================
    // RetinaFace Face Detector
    // ============================================================================

    class RetinaFace {
    public:
        RetinaFace(const std::string& model_path, const std::vector<std::string>& providers = { "CPUExecutionProvider" }) {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RetinaFace");

            session_options_ = std::make_unique<Ort::SessionOptions>();
            session_options_->SetIntraOpNumThreads(4);
            session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            for (const auto& provider : providers) {
                if (provider == "CUDAExecutionProvider") {
                    OrtCUDAProviderOptions cuda_options;
                    cuda_options.device_id = 0;
                    session_options_->AppendExecutionProvider_CUDA(cuda_options);
                    break;
                }
            }

#ifdef _WIN32
            session_ = std::make_unique<Ort::Session>(*env_, stringToWstring(model_path).c_str(), *session_options_);
#else
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
#endif

            Ort::AllocatorWithDefaultOptions allocator;
            input_name_strings_.resize(session_->GetInputCount());
            input_names_.resize(session_->GetInputCount());
            for (size_t i = 0; i < session_->GetInputCount(); i++) {
                input_name_strings_[i] = session_->GetInputNameAllocated(i, allocator).get();
                input_names_[i] = input_name_strings_[i].c_str();
            }

            output_name_strings_.resize(session_->GetOutputCount());
            output_names_.resize(session_->GetOutputCount());
            for (size_t i = 0; i < session_->GetOutputCount(); i++) {
                output_name_strings_[i] = session_->GetOutputNameAllocated(i, allocator).get();
                output_names_[i] = output_name_strings_[i].c_str();
            }
        }

        void prepare(int input_size = 640, float det_thresh = 0.5f, float nms_thresh = 0.4f) {
            input_size_ = input_size;
            det_thresh_ = det_thresh;
            nms_thresh_ = nms_thresh;
        }

        std::vector<FaceDetection> detect(const cv::Mat& image_bgr) {
            int orig_h = image_bgr.rows, orig_w = image_bgr.cols;
            float im_ratio = static_cast<float>(orig_h) / orig_w;
            float model_ratio = 1.0f;

            int new_w, new_h;
            if (im_ratio > model_ratio) {
                new_h = input_size_;
                new_w = static_cast<int>(new_h / im_ratio);
            }
            else {
                new_w = input_size_;
                new_h = static_cast<int>(new_w * im_ratio);
            }

            float det_scale = static_cast<float>(new_h) / static_cast<float>(orig_h);

            cv::Mat resized;
            cv::resize(image_bgr, resized, cv::Size(new_w, new_h));

            cv::Mat det_img(input_size_, input_size_, CV_8UC3, cv::Scalar(0, 0, 0));
            resized.copyTo(det_img(cv::Rect(0, 0, new_w, new_h)));

            cv::Mat rgb;
            cv::cvtColor(det_img, rgb, cv::COLOR_BGR2RGB);

            std::vector<float> input_data(3 * input_size_ * input_size_);
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < input_size_; h++) {
                    for (int w = 0; w < input_size_; w++) {
                        float val = static_cast<float>(rgb.at<cv::Vec3b>(h, w)[c]);
                        input_data[c * input_size_ * input_size_ + h * input_size_ + w] = (val - 127.5f) / 128.0f;
                    }
                }
            }

            std::vector<int64_t> input_shape = { 1, 3, input_size_, input_size_ };
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

            auto output_tensors = session_->Run(Ort::RunOptions{ nullptr }, input_names_.data(), &input_tensor, 1, output_names_.data(), output_names_.size());

            return forward(output_tensors, det_scale, orig_w, orig_h);
        }

    private:
        std::vector<float> distance2bbox(const std::vector<cv::Point2f>& anchors, const float* preds, int n) {
            std::vector<float> bboxes(n * 4);
            for (int i = 0; i < n; i++) {
                bboxes[i * 4 + 0] = anchors[i].x - preds[i * 4 + 0];
                bboxes[i * 4 + 1] = anchors[i].y - preds[i * 4 + 1];
                bboxes[i * 4 + 2] = anchors[i].x + preds[i * 4 + 2];
                bboxes[i * 4 + 3] = anchors[i].y + preds[i * 4 + 3];
            }
            return bboxes;
        }

        std::vector<float> distance2kps(const std::vector<cv::Point2f>& anchors, const float* preds, int n) {
            std::vector<float> kpss(n * 10);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < 5; j++) {
                    kpss[i * 10 + j * 2 + 0] = anchors[i].x + preds[i * 10 + j * 2 + 0];
                    kpss[i * 10 + j * 2 + 1] = anchors[i].y + preds[i * 10 + j * 2 + 1];
                }
            }
            return kpss;
        }

        std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, float thresh) {
            std::vector<int> order(scores.size()), indices;
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

            std::vector<bool> suppressed(scores.size(), false);
            for (size_t i = 0; i < order.size(); i++) {
                int idx = order[i];
                if (suppressed[idx]) continue;
                indices.push_back(idx);

                for (size_t j = i + 1; j < order.size(); j++) {
                    int jdx = order[j];
                    if (suppressed[jdx]) continue;

                    float x1 = std::max(boxes[idx].x, boxes[jdx].x);
                    float y1 = std::max(boxes[idx].y, boxes[jdx].y);
                    float x2 = std::min(boxes[idx].x + boxes[idx].width, boxes[jdx].x + boxes[jdx].width);
                    float y2 = std::min(boxes[idx].y + boxes[idx].height, boxes[jdx].y + boxes[jdx].height);

                    float inter = std::max(0.0f, x2 - x1 + 1) * std::max(0.0f, y2 - y1 + 1);
                    float area1 = (boxes[idx].width + 1) * (boxes[idx].height + 1);
                    float area2 = (boxes[jdx].width + 1) * (boxes[jdx].height + 1);
                    float iou = inter / (area1 + area2 - inter + 1e-6f);

                    if (iou > thresh) suppressed[jdx] = true;
                }
            }
            return indices;
        }

        std::vector<FaceDetection> forward(const std::vector<Ort::Value>& outputs, float det_scale, int orig_w, int orig_h) {
            std::vector<FaceDetection> detections;
            std::vector<std::vector<float>> scores_list, bboxes_list, kpss_list;

            constexpr int fmc = 3;
            constexpr int feat_stride_fpn[3] = { 8, 16, 32 };
            constexpr int num_anchors = 2;

            for (int idx = 0; idx < fmc; idx++) {
                int stride = feat_stride_fpn[idx];
                const float* scores_data = outputs[idx].GetTensorData<float>();
                const float* bbox_data = outputs[idx + fmc].GetTensorData<float>();
                const float* kps_data = outputs[idx + fmc * 2].GetTensorData<float>();

                int num = static_cast<int>(outputs[idx].GetTensorTypeAndShapeInfo().GetShape()[0]);
                int height = input_size_ / stride, width = input_size_ / stride;

                std::vector<cv::Point2f> anchors;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        for (int k = 0; k < num_anchors; k++) {
                            anchors.push_back(cv::Point2f(static_cast<float>(w) * stride, static_cast<float>(h) * stride));
                        }
                    }
                }

                std::vector<float> bbox_scaled(num * 4), kps_scaled(num * 10);
                for (int i = 0; i < num * 4; i++) bbox_scaled[i] = bbox_data[i] * stride;
                for (int i = 0; i < num * 10; i++) kps_scaled[i] = kps_data[i] * stride;

                auto bboxes = distance2bbox(anchors, bbox_scaled.data(), num);
                auto kpss = distance2kps(anchors, kps_scaled.data(), num);

                std::vector<float> scores;
                std::vector<int> pos_inds;
                for (int i = 0; i < num; i++) {
                    if (scores_data[i] >= det_thresh_) {
                        scores.push_back(scores_data[i]);
                        pos_inds.push_back(i);
                    }
                }

                std::vector<float> pos_bboxes, pos_kpss;
                for (int i : pos_inds) {
                    pos_bboxes.insert(pos_bboxes.end(), bboxes.begin() + i * 4, bboxes.begin() + i * 4 + 4);
                    pos_kpss.insert(pos_kpss.end(), kpss.begin() + i * 10, kpss.begin() + i * 10 + 10);
                }

                scores_list.push_back(scores);
                bboxes_list.push_back(pos_bboxes);
                kpss_list.push_back(pos_kpss);
            }

            std::vector<float> all_scores;
            std::vector<cv::Rect2f> all_boxes;
            std::vector<std::vector<float>> all_landmarks;

            for (size_t level = 0; level < scores_list.size(); level++) {
                for (size_t i = 0; i < scores_list[level].size(); i++) {
                    float x1 = bboxes_list[level][i * 4 + 0] / det_scale;
                    float y1 = bboxes_list[level][i * 4 + 1] / det_scale;
                    float x2 = bboxes_list[level][i * 4 + 2] / det_scale;
                    float y2 = bboxes_list[level][i * 4 + 3] / det_scale;

                    x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
                    y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
                    x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
                    y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

                    all_scores.push_back(scores_list[level][i]);
                    all_boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));

                    std::vector<float> lm(10);
                    for (int k = 0; k < 10; k++) {
                        lm[k] = kpss_list[level][i * 10 + k] / det_scale;
                    }
                    all_landmarks.push_back(lm);
                }
            }

            std::vector<int> order(all_scores.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return all_scores[a] > all_scores[b]; });

            std::vector<cv::Rect2f> sorted_boxes(all_boxes.size());
            std::vector<float> sorted_scores(all_scores.size());
            std::vector<std::vector<float>> sorted_landmarks(all_landmarks.size());
            for (size_t i = 0; i < order.size(); i++) {
                sorted_boxes[i] = all_boxes[order[i]];
                sorted_scores[i] = all_scores[order[i]];
                sorted_landmarks[i] = all_landmarks[order[i]];
            }

            std::vector<int> keep = nms(sorted_boxes, sorted_scores, nms_thresh_);

            for (int idx : keep) {
                FaceDetection det;
                det.bbox = sorted_boxes[idx];
                det.confidence = sorted_scores[idx];
                for (int k = 0; k < 5; k++) {
                    det.landmarks.points[k].x = sorted_landmarks[idx][k * 2 + 0];
                    det.landmarks.points[k].y = sorted_landmarks[idx][k * 2 + 1];
                }
                detections.push_back(det);
            }

            std::sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) {
                return a.bbox.area() > b.bbox.area();
            });

            return detections;
        }

    private:
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::Session> session_;
        std::unique_ptr<Ort::SessionOptions> session_options_;
        std::vector<const char*> input_names_, output_names_;
        std::vector<std::string> input_name_strings_, output_name_strings_;
        int input_size_ = 640;
        float det_thresh_ = 0.5f, nms_thresh_ = 0.4f;
    };

    // ============================================================================
    // Deepfake Detector
    // ============================================================================

    class DeepfakeDetector {
    public:
        DeepfakeDetector(const std::string& detector_path, const std::string& gend_path,
            const std::vector<std::string>& providers = { "CPUExecutionProvider" }) {
            detector_ = std::make_unique<RetinaFace>(detector_path, providers);
            detector_->prepare(640, det_thresh_, nms_thresh_);

            gend_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GenD");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(4);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            for (const auto& provider : providers) {
                if (provider == "CUDAExecutionProvider") {
                    OrtCUDAProviderOptions cuda_options;
                    cuda_options.device_id = 0;
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    break;
                }
            }

#ifdef _WIN32
            gend_session_ = std::make_unique<Ort::Session>(*gend_env_, stringToWstring(gend_path).c_str(), session_options);
#else
            gend_session_ = std::make_unique<Ort::Session>(*gend_env_, gend_path.c_str(), session_options);
#endif

            Ort::AllocatorWithDefaultOptions allocator;
            gend_input_name_strings_.resize(gend_session_->GetInputCount());
            gend_input_names_.resize(gend_session_->GetInputCount());
            for (size_t i = 0; i < gend_session_->GetInputCount(); i++) {
                gend_input_name_strings_[i] = gend_session_->GetInputNameAllocated(i, allocator).get();
                gend_input_names_[i] = gend_input_name_strings_[i].c_str();
            }

            gend_output_name_strings_.resize(gend_session_->GetOutputCount());
            gend_output_names_.resize(gend_session_->GetOutputCount());
            for (size_t i = 0; i < gend_session_->GetOutputCount(); i++) {
                gend_output_name_strings_[i] = gend_session_->GetOutputNameAllocated(i, allocator).get();
                gend_output_names_[i] = gend_output_name_strings_[i].c_str();
            }

            preprocessor_ = std::make_unique<Preprocessor>();
        }

        void setDetectionThreshold(float thresh) {
            det_thresh_ = thresh;
            detector_->prepare(640, det_thresh_, nms_thresh_);
        }

        void setFaceScale(float scale) { face_scale_ = scale; }
        void setMaxFaces(int max_faces) { max_faces_ = max_faces; }

        ImageResult detect(const std::string& image_path) {
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) throw std::runtime_error("Cannot read: " + image_path);
            ImageResult result = detect(image);
            result.image_path = image_path;
            return result;
        }

        ImageResult detect(const cv::Mat& image_bgr) {
            ImageResult result;
            auto detections = detector_->detect(image_bgr);

            if (max_faces_ > 0 && detections.size() > static_cast<size_t>(max_faces_)) {
                detections.resize(max_faces_);
            }

            for (const auto& det : detections) {
                cv::Mat aligned = alignFace(image_bgr, det.landmarks, face_scale_);
                cv::Mat rgb;
                cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);

                auto input_data = preprocessor_->process(rgb);
                auto input_shape = preprocessor_->getInputShape();

                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

                auto outputs = gend_session_->Run(Ort::RunOptions{ nullptr }, gend_input_names_.data(), &input_tensor, 1, gend_output_names_.data(), gend_output_names_.size());

                const float* logits = outputs[0].GetTensorData<float>();
                float max_val = std::max(logits[0], logits[1]);
                float exp0 = std::exp(logits[0] - max_val);
                float exp1 = std::exp(logits[1] - max_val);
                float fake_prob = exp1 / (exp0 + exp1);

                result.faces.push_back({ det.bbox, fake_prob });
            }

            if (!result.faces.empty()) {
                float sum = 0.0f;
                for (const auto& f : result.faces) sum += f.fake_prob;
                result.avg_fake_prob = sum / result.faces.size();
            }

            return result;
        }

        cv::Mat annotate(const cv::Mat& image_bgr, const ImageResult& result) {
            cv::Mat vis = image_bgr.clone();
            for (const auto& face : result.faces) {
                int x1 = static_cast<int>(face.bbox.x);
                int y1 = static_cast<int>(face.bbox.y);
                int x2 = static_cast<int>(face.bbox.x + face.bbox.width);
                int y2 = static_cast<int>(face.bbox.y + face.bbox.height);

                cv::Scalar color(0, static_cast<int>(255 * (1 - face.fake_prob)), static_cast<int>(255 * face.fake_prob));
                cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << face.fake_prob;
                std::string text = "fake: " + ss.str();

                int y_text = std::max(20, y1 + 20);
                cv::putText(vis, text, cv::Point(x1 + 6, y_text), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3);
                cv::putText(vis, text, cv::Point(x1 + 6, y_text), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            }
            return vis;
        }

    private:
        std::unique_ptr<RetinaFace> detector_;
        std::unique_ptr<Ort::Env> gend_env_;
        std::unique_ptr<Ort::Session> gend_session_;
        std::unique_ptr<Preprocessor> preprocessor_;
        std::vector<const char*> gend_input_names_, gend_output_names_;
        std::vector<std::string> gend_input_name_strings_, gend_output_name_strings_;
        float det_thresh_ = 0.5f, nms_thresh_ = 0.4f, face_scale_ = 1.3f;
        int max_faces_ = -1;
    };

    // ============================================================================
    // Utility Functions
    // ============================================================================

    void printResult(const ImageResult& result) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Image: " << result.image_path << std::endl;
        std::cout << "Faces: " << result.faces.size() << std::endl;
        for (size_t i = 0; i < result.faces.size(); i++) {
            std::cout << "  Face " << (i + 1) << ": Fake=" << std::fixed << std::setprecision(2)
                << (result.faces[i].fake_prob * 100.0f) << "%" << std::endl;
        }
        std::cout << "Average: Fake=" << std::fixed << std::setprecision(2)
            << (result.avg_fake_prob * 100.0f) << "%" << std::endl;
        std::cout << std::string(50, '=') << "\n" << std::endl;
    }

    void printUsage(const char* prog_name) {
        std::cout << "Deepfake Detection\n\n"
            << "Usage: " << prog_name << " --image <path> [options]\n\n"
            << "Options:\n"
            << "  --image <path>       Input image path (required)\n"
            << "  --detector <path>    Face detector model (default: det_10g.onnx)\n"
            << "  --gend <path>        GenD model (default: gend.onnx)\n"
            << "  --output <path>      Output image path (default: result.jpg)\n"
            << "  --thresh <float>     Detection threshold (default: 0.5)\n"
            << "  --scale <float>      Face alignment scale (default: 1.3)\n"
            << "  --max-faces <int>    Max faces to detect (default: unlimited)\n"
            << "  --gpu                Use GPU (CUDA)\n"
            << "  --help               Show this help\n"
            << std::endl;
    }

} // namespace deepfake

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string image_path="Shock.jpg";
    std::string detector_path = "det_10g.onnx", gend_path = "gend.onnx", output_path = "result.jpg";
    float det_thresh = 0.5f, face_scale = 1.3f;
    int max_faces = -1;
    bool use_gpu = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--detector" && i + 1 < argc) detector_path = argv[++i];
        else if (arg == "--gend" && i + 1 < argc) gend_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--thresh" && i + 1 < argc) det_thresh = std::stof(argv[++i]);
        else if (arg == "--scale" && i + 1 < argc) face_scale = std::stof(argv[++i]);
        else if (arg == "--max-faces" && i + 1 < argc) max_faces = std::stoi(argv[++i]);
        else if (arg == "--gpu") use_gpu = true;
        else if (arg == "--help") { deepfake::printUsage(argv[0]); return 0; }
        else { std::cerr << "Unknown argument: " << arg << std::endl; return 1; }
    }

    if (image_path.empty()) {
        std::cerr << "Error: --image is required" << std::endl;
        deepfake::printUsage(argv[0]);
        return 1;
    }

    try {
        auto providers = use_gpu ? std::vector<std::string>{"CUDAExecutionProvider", "CPUExecutionProvider"}
        : std::vector<std::string>{ "CPUExecutionProvider" };

        deepfake::DeepfakeDetector detector(detector_path, gend_path, providers);
        detector.setDetectionThreshold(det_thresh);
        detector.setFaceScale(face_scale);
        if (max_faces > 0) detector.setMaxFaces(max_faces);

        auto result = detector.detect(image_path);
        deepfake::printResult(result);

        cv::Mat image = cv::imread(image_path);
        cv::Mat annotated = detector.annotate(image, result);
        cv::imwrite(output_path, annotated);
        std::cout << "Output saved to: " << output_path << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
