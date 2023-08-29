#pragma once
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<unordered_map>
#include <ncnn/layer.h>
#include <ncnn/net.h>
#include <ncnn/benchmark.h>
#include<algorithm>

using namespace std;

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

class Model {
public:
    virtual bool Init(std::string modelPath) {
        return 1;
    }
    virtual vector<Object> Detect(cv::Mat bitmap, bool use_gpu) {
        return {};
    }
};

class ResNet : public Model{
public:

    bool Init(std::string modelPath) override {
        int ret1 = resNet50.load_param((modelPath + ".param").c_str());
        int ret2 = resNet50.load_model((modelPath + ".bin").c_str());
        std::cout << "模型地址: " << &resNet50 << '\n';
        if (ret1 && ret2)
            return true;
        else
            return false;
    }

    vector<Object> Detect(cv::Mat image, bool use_gpu) override {

        double start_time = ncnn::get_current_time();
        // ncnn from bitmap
        const int target_size = 224;

        cv::resize(image, image, { target_size, target_size });

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, 224, 224);

        float mean[3] = { 0.485 * 255.f, 0.456 * 255.f, 0.406 * 255.f };
        float std[3] = { 1.0 / 0.229 / 255, 1.0 / 0.224 / 255, 1.0 / 0.225 / 255 };
        in.substract_mean_normalize(mean, std);

        ncnn::Extractor ex = resNet50.create_extractor();

        ex.set_vulkan_compute(use_gpu);
        ex.input("input", in);
        ncnn::Mat preds;
        ex.extract("output", preds);
        float max_prob = 0.0f;
        int max_index = 0;
        for (int i = 0; i < preds.w; i++) {
            float prob = preds[i];
            if (prob > max_prob) {
                max_prob = prob;
                max_index = i;
            }
            //std::cout << "概率:" << prob << '\n';
        }
        vector<Object> ans;
        double end_time = ncnn::get_current_time() - start_time;
        std::cout << end_time << "ms Dectet\n";
        if (max_prob < 2)ans.push_back({ 10,10,0,0,preds.w, max_prob});
        else ans.push_back({ 10, 10, 0, 0, max_index, max_prob });
        return ans;
    }

private:
    ncnn::Net resNet50;

};

class Yolo :public Model{
public:

    static inline float intersection_area(const Object& a, const Object& b)
    {
        if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
        {
            // no intersection
            return 0.f;
        }

        float inter_width = min(a.x + a.w, b.x + b.w) - max(a.x, b.x);
        float inter_height = min(a.y + a.h, b.y + b.h) - max(a.y, b.y);

        return inter_width * inter_height;
    }

    static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);

                i++;
                j--;
            }
        }

#pragma omp parallel sections
        {
#pragma omp section
            {
                if (left < j) qsort_descent_inplace(faceobjects, left, j);
            }
#pragma omp section
            {
                if (i < right) qsort_descent_inplace(faceobjects, i, right);
            }
        }
    }

    static void qsort_descent_inplace(std::vector<Object>& faceobjects)
    {
        if (faceobjects.empty())
            return;

        qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
    }

    static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].w * faceobjects[i].h;
        }

        for (int i = 0; i < n; i++)
        {
            const Object& a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const Object& b = faceobjects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

    static inline float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
    {
        const int num_grid = feat_blob.h;

        int num_grid_x;
        int num_grid_y;
        if (in_pad.w > in_pad.h)
        {
            num_grid_x = in_pad.w / stride;
            num_grid_y = num_grid / num_grid_x;
        }
        else
        {
            num_grid_y = in_pad.h / stride;
            num_grid_x = num_grid / num_grid_y;
        }

        const int num_class = feat_blob.w - 5;

        const int num_anchors = anchors.w / 2;

        for (int q = 0; q < num_anchors; q++)
        {
            const float anchor_w = anchors[q * 2];
            const float anchor_h = anchors[q * 2 + 1];

            const ncnn::Mat feat = feat_blob.channel(q);

            for (int i = 0; i < num_grid_y; i++)
            {
                for (int j = 0; j < num_grid_x; j++)
                {
                    const float* featptr = feat.row(i * num_grid_x + j);

                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    float box_score = featptr[4];

                    float confidence = sigmoid(box_score) * sigmoid(class_score);

                    if (confidence >= prob_threshold)
                    {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Object obj;
                        obj.x = x0;
                        obj.y = y0;
                        obj.w = x1 - x0;
                        obj.h = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }

    bool Init(std::string modelName) override
    {
        ncnn::Option opt;
        opt.lightmode = true;
        opt.num_threads = 4;
        opt.blob_allocator = &g_blob_pool_allocator;
        opt.workspace_allocator = &g_workspace_pool_allocator;
        opt.use_packing_layout = true;

        std::string modelPath = modelName;

        // use vulkan compute
        if (ncnn::get_gpu_count() != 0)
            opt.use_vulkan_compute = true;

        yolov5.opt = opt;


        // init param
        {
            int ret = yolov5.load_param((modelPath + ".param").c_str());
            if (ret != 0)
            {
                std::cout <<  "YoloV5Ncnn load_param failed";
                return false;
            }
        }

        // init bin
        {
            int ret = yolov5.load_model((modelPath + ".bin").c_str());
            if (ret != 0)
            {
                std::cout<< "YoloV5Ncnn  load_model failed";
                return false;
            }
        }

        std::cout << "Yolov5 模型已加载\n";
        std::cout << "模型地址: " << &yolov5 << '\n';
        
        return true;
    }

    // public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
    vector<Object> Detect(cv::Mat bitmap, bool use_gpu) override
    {
        if (use_gpu == true && ncnn::get_gpu_count() == 0)
        {
            return {};
        }
        double start_time = ncnn::get_current_time();

        const int width = bitmap.cols;
        const int height = bitmap.rows;


        // ncnn from bitmap
        const int target_size = 640;

        // letterbox pad to multiple of 32
        int w = width;
        int h = height;
        float scale = 1.f;
        if (w > h)
        {
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bitmap.data, ncnn::Mat::PIXEL_BGR, bitmap.cols, bitmap.rows, w, h);

        // pad to target_size rectangle
        // yolov5/utils/datasets.py letterbox
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        // yolov5
        std::vector<Object> objects;
        {
            const float prob_threshold = 0.25f;
            const float nms_threshold = 0.45f;

            const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
            in_pad.substract_mean_normalize(0, norm_vals);

            ncnn::Extractor ex = yolov5.create_extractor();

            ex.set_vulkan_compute(use_gpu);

            ex.input("images", in_pad);

            std::vector<Object> proposals;

            // anchor setting from yolov5/models/yolov5s.yaml

            // stride 8
            {
                ncnn::Mat out;
                ex.extract("output", out);

                ncnn::Mat anchors(6);
                anchors[0] = 10.f;
                anchors[1] = 13.f;
                anchors[2] = 16.f;
                anchors[3] = 30.f;
                anchors[4] = 33.f;
                anchors[5] = 23.f;

                std::vector<Object> objects8;
                generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

                proposals.insert(proposals.end(), objects8.begin(), objects8.end());
            }
            // stride 16
            {
                ncnn::Mat out;
                ex.extract("471", out);

                ncnn::Mat anchors(6);
                anchors[0] = 30.f;
                anchors[1] = 61.f;
                anchors[2] = 62.f;
                anchors[3] = 45.f;
                anchors[4] = 59.f;
                anchors[5] = 119.f;

                std::vector<Object> objects16;
                generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

                proposals.insert(proposals.end(), objects16.begin(), objects16.end());
            }

            // stride 32
            {
                ncnn::Mat out;
                ex.extract("483", out);

                ncnn::Mat anchors(6);
                anchors[0] = 116.f;
                anchors[1] = 90.f;
                anchors[2] = 156.f;
                anchors[3] = 198.f;
                anchors[4] = 373.f;
                anchors[5] = 326.f;

                std::vector<Object> objects32;
                generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

                proposals.insert(proposals.end(), objects32.begin(), objects32.end());
            }

            // sort all proposals by score from highest to lowest
            qsort_descent_inplace(proposals);

            // apply nms with nms_threshold
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_threshold);

            int count = picked.size();

            objects.resize(count);
            for (int i = 0; i < count; i++)
            {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = (objects[i].x - (wpad / 2)) / scale;
                float y0 = (objects[i].y - (hpad / 2)) / scale;
                float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
                float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

                // clip
                x0 = max(min(x0, (float)(width - 1)), 0.f);
                y0 = max(min(y0, (float)(height - 1)), 0.f);
                x1 = max(min(x1, (float)(width - 1)), 0.f);
                y1 = max(min(y1, (float)(height - 1)), 0.f);

                objects[i].x = x0;
                objects[i].y = y0;
                objects[i].w = x1 - x0;
                objects[i].h = y1 - y0;
            }
        }

        /*
        // objects to Obj[]
        static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };
         */

        double elasped = ncnn::get_current_time() - start_time;
        std::cout<<  "YoloV5Ncnn " << elasped << "ms   detect\n";

        return objects;
    }

private:
    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;
    ncnn::Net yolov5;

};