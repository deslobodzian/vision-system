#include "inference/yolo.hpp"
#include "NvOnnxParser.h"

using namespace nvinfer1;

static Logger gLogger;

Yolo::Yolo() {

}
Yolo::~Yolo() {
    if (is_init_) {
        cudaStreamDestroy(stream_);
        CUDA_CHECK(cudaFree(d_input_));
        CUDA_CHECK(cudaFree(d_output_));

        context_->destroy();
        engine_->destroy();
        runtime_->destroy();

        delete[] h_input_;
        delete[] h_output_;
    }
    is_init_ = false;
}

int Yolo::build_engine(std::string onnx_path, std::string engine_path, OptimDim dyn_dim_profile) {
    std::vector<uint8_t> onnx_file_content;
    if (readFile(onnx_path, onnx_file_content)) return 1;

    if ((!onnx_file_content.empty())) {

        ICudaEngine * engine;
        // Create engine (onnx)
        std::cout << "Creating engine from onnx model" << std::endl;

        gLogger.setReportableSeverity(Severity::kINFO);
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            std::cerr << "createInferBuilder failed" << std::endl;
            return 1;
        }

        auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicitBatch);

        if (!network) {
            std::cerr << "createNetwork failed" << std::endl;
            return 1;
        }

        auto config = builder->createBuilderConfig();
        if (!config) {
            std::cerr << "createBuilderConfig failed" << std::endl;
            return 1;
        }

        ////////// Dynamic dimensions handling : support only 1 size at a time
        if (!dyn_dim_profile.tensor_name.empty()) {

            IOptimizationProfile* profile = builder->createOptimizationProfile();

            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kMIN, dyn_dim_profile.size);
            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kOPT, dyn_dim_profile.size);
            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kMAX, dyn_dim_profile.size);

            config->addOptimizationProfile(profile);
            builder->setMaxBatchSize(1);
        }

        auto parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser) {
            std::cerr << "nvonnxparser::createParser failed" << std::endl;
            return 1;
        }

        bool parsed = false;
        unsigned char *onnx_model_buffer = onnx_file_content.data();
        size_t onnx_model_buffer_size = onnx_file_content.size() * sizeof (char);
        parsed = parser->parse(onnx_model_buffer, onnx_model_buffer_size);

        if (!parsed) {
            std::cerr << "onnx file parsing failed" << std::endl;
            return 1;
        }

        if (builder->platformHasFastFp16()) {
            std::cout << "FP16 enabled!" << std::endl;
            config->setFlag(BuilderFlag::kFP16);
        }

        //////////////// Actual engine building

        engine = builder->buildEngineWithConfig(*network, *config);

        onnx_file_content.clear();

        // write plan file if it is specified        
        if (engine == nullptr) return 1;
        IHostMemory* ptr = engine->serialize();
        assert(ptr);
        if (ptr == nullptr) return 1;

        FILE *fp = fopen(engine_path.c_str(), "wb");
        fwrite(reinterpret_cast<const char*> (ptr->data()), ptr->size() * sizeof (char), 1, fp);
        fclose(fp);

        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();

        engine->destroy();

        return 0;
    } else return 1;
}
int Yolo::init(std::string engine_name) {
    std::ifstream file(engine_name, std::ios::binary);

    if (!file.good()) {
        std::cerr << "[Error] read " <<engine_name << " failed!\n";
        return -1;
    }
    info("Read engine name: "+ engine_name + " suceesfully");

    char *trt_model_stream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    if (!trt_model_stream) return 1;
    file.read(trt_model_stream, size);
    file.close();

    runtime_ = createInferRuntime(gLogger);
    if (runtime_ == nullptr) return 1;
    engine_ = runtime_->deserializeCudaEngine(trt_model_stream, size);
    if (engine_ == nullptr) return 1;
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) return 1;

    const int bindings = engine_->getNbBindings();
    for (int i = 0; i < bindings; i++) {
        if (engine_->bindingIsInput(i)) {
            input_binding_name_ = engine_->getBindingName(i);
            Dims bind_dim = engine_->getBindingDimensions(i);
            input_width_ = bind_dim.d[3];
            input_height_ = bind_dim.d[2];
            input_index_ = i;
            info("Inference size: " + std::to_string(input_height_) + "x" + std::to_string(input_width_) + "\n");
        } else {
            output_name_ = engine_->getBindingName(i);
            output_index_ = i;
            Dims bind_dim = engine_->getBindingDimensions(i);
            size_t batch = bind_dim.d[0];
            if (batch > batch_size_) {
            std::cout << "batch > 1 not supported" << std::endl;
            return 1;
        }
        size_t dim1 = bind_dim.d[1];
        size_t dim2 = bind_dim.d[2];

        // Yolov8 1x84x8400
        out_dim_ = dim2;
        out_box_struct_number_ = 4;
        out_class_number_ = dim1 - out_box_struct_number_;
        std::cout << "YOLOV8/YOLOV5 format" << std::endl;
        }
    }
    output_size_ = out_dim_ * (out_class_number_ + out_box_struct_number_);
    h_input_ = new float[batch_size_ * 3 * input_height_ * input_width_];
    h_output_ = new float[batch_size_ * output_size_];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    assert(input_index_ == 0);
    assert(output_index_ == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&d_input_, batch_size_ * 3 * input_height_ * input_width_ * sizeof (float)));
    CUDA_CHECK(cudaMalloc(&d_output_, batch_size_ * output_size_ * sizeof (float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream_));

    if (batch_size_ != 1) return 1; // This sample only support batch 1 for now

    is_init_ = true;
    return 0;
}

std::vector<BBoxInfo> Yolo::run(sl::Mat left_img, int orig_image_h, int orig_image_w, float thres) {
    std::vector<BBoxInfo> binfo;

    size_t frame_s = input_height_ * input_width_;

    cv::Mat left_cv_rgba = slMat_to_cvMat(left_img);
    cv::cvtColor(left_cv_rgba, left_cv_rgb_, cv::COLOR_BGRA2BGR);
    if (left_cv_rgb_.empty()) return binfo;
    cv::Mat pr_img = preprocess_img(left_cv_rgb_, input_width_, input_height_); // letterbox BGR to RGB
    int i = 0;
    int batch = 0;
    for (int row = 0; row < input_height_; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < input_width_; ++col) {
            h_input_[batch * 3 * frame_s + i] = (float) uc_pixel[2] / 255.0;
            h_input_[batch * 3 * frame_s + i + frame_s] = (float) uc_pixel[1] / 255.0;
            h_input_[batch * 3 * frame_s + i + 2 * frame_s] = (float) uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

     /////// INFERENCE
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input_, batch_size_ * 3 * frame_s * sizeof (float), cudaMemcpyHostToDevice, stream_));

    std::vector<void*> d_buffers_nvinfer(2);
    d_buffers_nvinfer[input_index_] = d_input_;
    d_buffers_nvinfer[output_index_] = d_output_;
    context_->enqueueV2(&d_buffers_nvinfer[0], stream_, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(h_output_, d_output_, batch_size_ * output_size_ * sizeof (float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
    
    float scalingFactor = std::min(static_cast<float> (input_width_) / orig_image_w, static_cast<float> (input_height_) / orig_image_h);
    float xOffset = (input_width_ - scalingFactor * orig_image_w) * 0.5f;
    float yOffset = (input_height_ - scalingFactor * orig_image_h) * 0.5f;
    scalingFactor = 1.f / scalingFactor;
    float scalingFactor_x = scalingFactor;
    float scalingFactor_y = scalingFactor;

    auto num_channels = out_class_number_ + out_box_struct_number_;
    auto num_anchors = out_dim_;
    auto num_labels = out_class_number_;

    auto& dw = xOffset;
    auto& dh = yOffset;

    auto& width = orig_image_w;
    auto& height = orig_image_h;

    cv::Mat output = cv::Mat(
            num_channels,
            num_anchors,
            CV_32F,
            static_cast<float*> (h_output_)
            );
    output = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + out_box_struct_number_;
        auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score = *max_s_ptr;
        if (score > thres) {
            int label = max_s_ptr - scores_ptr;

            BBoxInfo bbi;

            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * scalingFactor_x, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * scalingFactor_y, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * scalingFactor_x, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * scalingFactor_y, 0.f, height);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bbi.box.x1 = x0;
            bbi.box.y1 = y0;
            bbi.box.x2 = x1;
            bbi.box.y2 = y1;

            if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2)) break;

            bbi.label = label;
            bbi.prob = score;

            binfo.push_back(bbi);
        }
    }
    binfo = non_maximum_suppression(nms_, binfo);
    return binfo;
}

