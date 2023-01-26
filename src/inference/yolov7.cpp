#include "inference/yolov7.hpp"

std::vector<sl::uint2> Yolov7::cvt(const cv::Rect& bbox_in){
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}
void Yolov7::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
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
    deserialize_engine(engine_name, &runtime_, &engine_, &context_);
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream_));
    assert(kBatchSize == 1); // This sample only support batch 1 for now

    cuda_preprocess_init(kMaxInputImageSize);

    prepare_buffer(engine_, &buffers_[0], &buffers_[1], &output_buffer_host_);
    return false;
}

void Yolov7::doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchSize) {
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
    nms(res, &output_buffer_host_[batch_ * kOutputSize], kConfThresh, kNmsThresh);
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
void Yolov7::run_inference_test(cv::Mat& img_cv_rgb) {
    auto start = std::chrono::high_resolution_clock::now();
    doInference(*context_, stream_, (void **)buffers_, output_buffer_host_, kBatchSize);
    auto stop = std::chrono::high_resolution_clock::now();
    info("Inference time: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) + "ms");
    std::vector<std::vector<Detection>> batch_res(kBatchSize);
    auto& res = batch_res[batch_];
    nms(res, &output_buffer_host_[batch_ * kOutputSize], kConfThresh, kNmsThresh);
    draw_bbox_single(img_cv_rgb, res);
}

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



