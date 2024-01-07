#ifdef __CUDACC__
#include "inference/tensor_rt_engine.hpp"
#include "

static Logger gLogger;

TensorRTEngine::TensorRTEngine() {

}

TensorRTEngine::~TensorRTEngine() {

}

int TensorRTEngine::build_engine(std::string onnx_path, std::string engine_path, OptimDim dyn_dim_profile) {
    std::vector<uint8_t> onnx_file_content;
    if (readFile(onnx_path, onnx_file_content)) return 1;

    if ((!onnx_file_content.empty())) {

        ICudaEngine * engine;
        // Create engine (onnx)
        LOG_INFO("Creating engine from onnx model");

        gLogger.setReportableSeverity(Severity::kINFO);
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            LOG_ERROR("createInferBuilder failed");
            return 1;
        }

        auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicitBatch);

        if (!network) {
            LOG_ERROR("createNetwork failed");
            return 1;
        }

        auto config = builder->createBuilderConfig();
        if (!config) {
            LOG_ERROR("createBuilderConfig failed");
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
            LOG_ERROR("nvonnxparser::createParser failed");
            return 1;
        }

        bool parsed = false;
        unsigned char *onnx_model_buffer = onnx_file_content.data();
        size_t onnx_model_buffer_size = onnx_file_content.size() * sizeof (char);
        parsed = parser->parse(onnx_model_buffer, onnx_model_buffer_size);

        if (!parsed) {
            LOG_ERROR("onnx file parsing failed");
            return 1;
        }

        if (builder->platformHasFastFp16()) {
            LOG_INFO("FP16 enabled!");
            config->setFlag(BuilderFlag::kFP16);
        }

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

int TesorRTEngine::load_model(const std::string& model_path) {
    std::ifstream file(engine_name, std::ios::binary);

    if (!file.good()) {
        LOG_ERROR("read ", engine_name, " failed!";
                return -1;
                }
                LOG_INFO("Read engine name: ", engine_name, " suceesfully");

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
                        LOG_INFO("Inference size: ", input_height_, "x", input_width_);
                    } else {
                        output_name_ = engine_->getBindingName(i);
                        output_index_ = i;
                        Dims bind_dim = engine_->getBindingDimensions(i);
                        size_t batch = bind_dim.d[0];
                        if (batch > batch_size_) {
                            LOG_INFO("batch > 1 not supported");
                            return 1;
                        }
                        size_t dim1 = bind_dim.d[1];
                        size_t dim2 = bind_dim.d[2];
                    }
                }
                output_size_ = out_dim_ * (out_class_number_ + out_box_struct_number_);

                input_tensor_ = Tensor<float>({batch_size_, 3, input_height_, input_width_}, Device::CPU);
                output_tensor_ = Tensor<float>({batch_size_, output_size_}, Device::CPU);

                //h_input_ = new float[batch_size_ * 3 * input_height_ * input_width_];
                //h_output_ = new float[batch_size_ * output_size_];

                assert(input_index_ == 0);
                assert(output_index_ == 1);
                // Create GPU buffers on device

                //CUDA_CHECK(cudaMalloc(&d_input_, batch_size_ * 3 * input_height_ * input_width_ * sizeof (float)));
                //CUDA_CHECK(cudaMalloc(&d_output_, batch_size_ * output_size_ * sizeof (float)));
                /
                // Create stream
                CUDA_CHECK(cudaStreamCreate(&stream_));

                if (batch_size_ != 1) return 1; // This sample only support batch 1 for now

                is_init_ = true;
                return 0;
}

#endif

