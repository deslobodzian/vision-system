#include "inference/yolov7.hpp"

std::vector<sl::uint2> Yolov7::cvt(const cv::Rect& bbox_in){
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}

void Yolov7::prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device, float** output_buffer_host) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));

    *output_buffer_host = new float[kBatchSize * kOutputSize];
}

bool Yolov7::initialize_engine(std::string& engine_name) {
	std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        error("[Yolov7]: Read " + engine_name + " error!");
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // prepare input data ---------------------------
    runtime_ = createInferRuntime(gLogger);
    assert(runtime_ != nullptr);
    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
    delete[] trtModelStream;
    assert(engine_->getNbBindings() == 2);

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream_));
    assert(kBatchSize == 1); // This sample only support batch 1 for now

    cuda_preprocess_init(kMaxInputImageSize);

    prepare_buffer(engine_, &buffers_[0], &buffers_[1], &output_buffer_host_);
    return false;
}

void Yolov7::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool Yolov7::prepare_inference(cv::Mat& img_cv_rgb) {
    cuda_preprocess(
            img_cv_rgb.ptr(),
            img_cv_rgb.cols,
            img_cv_rgb.rows,
            buffers_[0],
            kInputW,
            kInputH,
            stream_
            );
//    if (img_cv_rgb.empty()) return false;
//    cv::Mat pr_img = preprocess_img(img_cv_rgb, kInputW, kInputH);
//    int i = 0;
//    for (int row = 0; row < kInputH; ++row) {
//        uchar* uc_pixel = pr_img.data + row * pr_img.step;
//        for (int col = 0; col < kInputW; ++col) {
//            data[batch_ * 3 * kInputH * kInputW + i] = (float) uc_pixel[2] / 255.0;
//            data[batch_ * 3 * kInputH * kInputW + i + kInputH * kInputW] = (float) uc_pixel[1] / 255.0;
//            data[batch_ * 3 * kInputH * kInputW+ i + 2 * kInputH * kInputW] = (float) uc_pixel[0] / 255.0;
//            uc_pixel += 3;
//            ++i;
//        }
//    }
    return false;
}
bool Yolov7::prepare_inference(sl::Mat& img_sl, cv::Mat& img_cv_rgb) {
    cv::Mat left_cv_rgba = slMat_to_cvMat(img_sl);
    cv::cvtColor(left_cv_rgba, img_cv_rgb, cv::COLOR_BGRA2BGR);
    return prepare_inference(img_cv_rgb);
}
//
void Yolov7::run_inference(cv::Mat& img_cv_rgb, std::vector<sl::CustomBoxObjectData>* objs) {
    objs->clear();
//    auto start = std::chrono::high_resolution_clock::now();
    doInference(*context_, stream_, (void**)buffers_, output_buffer_host_, kBatchSize);
//    auto stop = std::chrono::high_resolution_clock::now();
//    info("Inference time: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) + "ms");
    std::vector<std::vector<Detection>> batch_res(kBatchSize);
    auto& res = batch_res[batch_];
    nms(res, &prob[batch_ * kOutputSize], kConfThresh, kNmsThresh);
    cv::Rect bounds = cv::Rect(0, 0, img_cv_rgb.size().width, img_cv_rgb.size().height);
    for (auto &it : res) {
	    sl::CustomBoxObjectData tmp;
	    cv::Rect r = get_rect(img_cv_rgb, it.bbox);
        // make sure our detected object bounds fit within the image frame.
        r = bounds & r;
	    tmp.unique_object_id = sl::generate_unique_id();
	    tmp.probability = it.conf;
	    tmp.label = (int) it.class_id;
	    tmp.bounding_box_2d = cvt(r);
	    objs->push_back(tmp);
	}
}
//
//void Yolov5::run_inference(cv::Mat& img_cv_rgb) {
//    monocular_objects_in_.clear();
//    doInference(*context_, stream_, buffers_, data, prob, BATCH_SIZE);
//    std::vector<std::vector<Yolo::Detection>> batch_res(BATCH_SIZE);
//    auto& res = batch_res[batch_];
//    nms(res, &prob[batch_ * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
//    for (auto &it : res) {
//        cv::Rect r = get_rect(img_cv_rgb, it.bbox);
//        tracked_object temp(r, it.class_id);
//        monocular_objects_in_.push_back(temp);
//    }
//}
//
//std::vector<sl::CustomBoxObjectData> Yolov5::get_custom_obj_data() {
//	return objects_in_;
//}
Yolov7::~Yolov7() {
    kill();
}
//
void Yolov7::kill() {
	CUDA_CHECK(cudaFree(buffers_[0]))
	CUDA_CHECK(cudaFree(buffers_[1]))
    delete[] output_buffer_host_;
    cuda_preprocess_destroy();
    delete context_;
    delete engine_;
	delete runtime_;
}



