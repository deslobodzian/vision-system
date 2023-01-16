#include "inference/yolov7.hpp"
//
//
//int Yolov5::get_width(int x, float gw, int divisor) {
//    return int(ceil((x * gw) / divisor)) * divisor;
//}
//
//int Yolov5::get_depth(int x, float gd) {
//    if (x == 1) return 1;
//    int r = round(x * gd);
//    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
//        --r;
//    }
//    return std::max<int>(r, 1);
//}
//
//
//std::vector<sl::uint2> Yolov5::cvt(const cv::Rect &bbox_in){
//    std::vector<sl::uint2> bbox_out(4);
//    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
//    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
//    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
//    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
//    return bbox_out;
//}
//

bool Yolov7::initialize_engine(std::string& engine_name) {
	std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
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
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], kBatchSize * 3 * kInputH * kInputW * sizeof (float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], kBatchSize * kOutputSize * sizeof (float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    assert(kBatchSize == 1); // This sample only support batch 1 for now
    return 0;
}

void Yolov7::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * kInputH * kInputW* sizeof (float), cudaMemcpyHostToDevice, stream_));
    if(!context.enqueue(batchSize, buffers, stream_, nullptr)) {
//        error("TensorRT Context error");
    }
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * kOutputSize * sizeof (float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
}

bool Yolov7::prepare_inference(cv::Mat& img_cv_rgb) {
    if (img_cv_rgb.empty()) return false;
    cv::Mat pr_img = preprocess_img(img_cv_rgb, kInputW, kInputH);
    int i = 0;
    for (int row = 0; row < kInputH; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < kInputW; ++col) {
            data[batch_ * 3 * kInputH * kInputW + i] = (float) uc_pixel[2] / 255.0;
            data[batch_ * 3 * kInputH * kInputW + i + kInputH * kInputW] = (float) uc_pixel[1] / 255.0;
            data[batch_ * 3 * kInputH * kInputW+ i + 2 * kInputH * kInputW] = (float) uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    return 0;
}
bool Yolov7::prepare_inference(sl::Mat& img_sl, cv::Mat& img_cv_rgb) {
    cv::Mat left_cv_rgba = slMat_to_cvMat(img_sl);
    cv::cvtColor(left_cv_rgba, img_cv_rgb, cv::COLOR_BGRA2BGR);
    return prepare_inference(img_cv_rgb);
}
//
//void Yolov5::run_inference_and_convert_to_zed(cv::Mat& img_cv_rgb) {
//    objects_in_.clear();
//    //auto start = std::chrono::high_resolution_clock::now();
//    doInference(*context_, stream_, buffers_, data, prob, BATCH_SIZE);
//    //auto stop = std::chrono::high_resolution_clock::now();
//    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
//    std::vector<std::vector<Yolo::Detection>> batch_res(BATCH_SIZE);
//    //info("Time takes: " + std::to_string(duration.count()));
//    auto& res = batch_res[batch_];
//    nms(res, &prob[batch_ * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
//    cv::Rect bounds = cv::Rect(0, 0, img_cv_rgb.size().width, img_cv_rgb.size().height);
//    for (auto &it : res) {
//	    sl::CustomBoxObjectData tmp;
//	    cv::Rect r = get_rect(img_cv_rgb, it.bbox);
//        // make sure our detected object bounds fit within the image frame.
//        r = bounds & r;
//	    tmp.unique_object_id = sl::generate_unique_id();
//	    tmp.probability = it.conf;
//	    tmp.label = (int) it.class_id;
//	    tmp.bounding_box_2d = cvt(r);
//	    objects_in_.push_back(tmp);
//	}
//}
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
//
//std::vector<tracked_object> Yolov5::get_monocular_obj_data() {
//    return monocular_objects_in_;
//}
//Yolov5::~Yolov5() {
//    kill();
//}
//
//void Yolov5::kill() {
//	CUDA_CHECK(cudaFree(buffers_[inputIndex_]))
//	CUDA_CHECK(cudaFree(buffers_[outputIndex_]))
//	context_->destroy();
//	engine_->destroy();
//	runtime_->destroy();
//}



