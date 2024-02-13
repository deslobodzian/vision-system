#ifdef WITH_CUDA
#include "inference/tensor_rt_engine.hpp"
#include "utils/logger.hpp"
#include "inference/int8_entropy_calibrator.hpp"
#include "inference/trt_logger.h"
#include <chrono>

static Logger gLogger;
using namespace nvinfer1;
using namespace nvonnxparser;

TensorRTEngine::TensorRTEngine() {}

TensorRTEngine::~TensorRTEngine() {
   delete context_;
   delete engine_;
   delete runtime_; 
   if (graph_initialized_) {
        CUDA_CHECK(cudaGraphExecDestroy(instance_));
        CUDA_CHECK(cudaGraphDestroy(graph_));
   }
}

void TensorRTEngine::set_execution_data(void* execution_data) {
    LOG_INFO("Setting cuda stream externally");
    stream_ = static_cast<cudaStream_t>(execution_data);
}

int TensorRTEngine::build_engine(const EngineConfig& cfg, OptimDim dyn_dim_profile) {
	std::vector<uint8_t> onnx_file_content;
	if (readFile(cfg.onnx_path, onnx_file_content)) return -1;
    if (onnx_file_content.empty()) return -1;

	LOG_INFO("Creating engine from onnx model");

	gLogger.setReportableSeverity(Severity::kINFO);
    IBuilder* builder = createInferBuilder(gLogger);
    if (!builder) {
        LOG_ERROR("createInferBuilder failed");
        return -1;
    }

    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);

    if (!network) {
        LOG_ERROR("createNetwork failed");
        delete builder;
        return -1;
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    if (!config){ 
        LOG_ERROR("createBuidlerConfig failed");
        delete network;
        delete builder;
        return -1;
    }

    if (!dyn_dim_profile.tensor_name.empty()) {

        IOptimizationProfile* profile = builder->createOptimizationProfile();

        profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kMIN, dyn_dim_profile.size);
        profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kOPT, dyn_dim_profile.size);
        profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kMAX, dyn_dim_profile.size);

        config->addOptimizationProfile(profile);
    }

    auto parser = createParser(*network, gLogger);

    if (!parser) {
        LOG_ERROR("createParser failed");
        delete config;
        delete network;
        delete builder;
        return -1;
    }

    bool parsed = parser->parseFromFile(cfg.onnx_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    if (!parsed) {
        LOG_ERROR("parseFromFile failed");
        for (int32_t i = 0; i < parser->getNbErrors(); i++) {
            LOG_ERROR(parser->getError(i)->desc()); 
        }
        delete parser;
        delete config;
        delete network;
        delete builder;
        return -1;
    }

    std::unique_ptr<Int8EntropyCalibrator2> calibrator_ = nullptr;
    switch(cfg.presicion) {
        case ModelPrecision::FP_32:
            LOG_INFO("Using FP_32");
            break;
        case ModelPrecision::FP_16:
            if (builder->platformHasFastFp16()) {
                LOG_INFO("Using FP_16");
                config->setFlag(BuilderFlag::kFP16);
            }
            break;
        case ModelPrecision::INT_8:
            std::string data = cfg.int8_data_path; //"/home/odin/Data/val2017";
            if (builder->platformHasFastInt8()) {
                LOG_INFO("Using INT_8");
                if (data.empty()) {
                    throw std::runtime_error("Error: If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
                }
                config->setFlag((BuilderFlag::kINT8));

                const auto input = network->getInput(0);
                const auto input_name = input->getName();
                const auto bind_dim = input->getDimensions();
                Shape input_shape({bind_dim.d[0], bind_dim.d[1], bind_dim.d[2], bind_dim.d[3]});
                const auto calibration_file_name = remove_file_extension(cfg.engine_path) + ".calibration";
                LOG_DEBUG(calibration_file_name);
                calibrator_ = std::make_unique<Int8EntropyCalibrator2>(1, data, calibration_file_name, input_name, input_shape);

                config->setInt8Calibrator(calibrator_.get());
            }
            break;
    }

    int dla_cores_available = builder->getNbDLACores();
    if (dla_cores_available <= 0) {
        LOG_INFO("DLA cores not available, using GPU");
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
    } else {
        LOG_INFO("DLA Cores Available, using ", dla_cores_available, " cores.");
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(dla_cores_available);
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
    }

    // Not in TRT 8.5
    //LOG_INFO("Setting optimization level: ", cfg.optimization_level);
    //config->setBuilderOptimizationLevel(cfg.optimization_level);

    LOG_INFO("Setting max threads: ", cfg.max_threads);
    builder->setMaxThreads(cfg.max_threads);

    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    if (!serialized_model) {
        LOG_ERROR("buildSerializedNetwork failed");
        delete parser;
        delete config;
        delete network;
        delete builder;
        return -1;
    }

    std::ofstream engine_file(cfg.engine_path, std::ios::binary);
    engine_file.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
    engine_file.close();
    delete serialized_model;
    delete parser;
    delete network;
    delete config;
    delete builder;
    return 0;
}

void TensorRTEngine::load_model(const std::string& model_path) {
    // I should do a check to make sure the engine exists later.
    std::string engine_model = remove_file_extension(model_path) + ".engine";
    
	std::ifstream file(engine_model, std::ios::binary);

	if (!file.good()) {
		LOG_ERROR("read ", model_path, " failed!");
	}
	LOG_INFO("Read engine name: ", model_path, " suceesfully");

	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
    std::vector<char> trt_model_stream(size);
//if (!trt_model_stream) LOG_ERROR("trt_model_stream not created");
	file.read(trt_model_stream.data(), size);
	file.close();
	runtime_ = createInferRuntime(gLogger);
	if (runtime_ == nullptr) LOG_ERROR("runtime not created");
	engine_ = runtime_->deserializeCudaEngine(trt_model_stream.data(), size);
	if (engine_ == nullptr) LOG_ERROR("engine not created");
	context_ = engine_->createExecutionContext();
	if (context_ == nullptr) LOG_ERROR("context not created");

	const int num_tensors = engine_->getNbIOTensors();
	for (int i = 0; i < num_tensors; i++) {
        const char* tensor_name = engine_->getIOTensorName(i);
        TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);

        if (io_mode == TensorIOMode::kINPUT) {
            input_name_ = tensor_name;
            LOG_INFO("Tensor RT Input name: ", input_name_);
            Dims bind_dim = engine_->getTensorShape(tensor_name);
            input_shape_ = {bind_dim.d[0], bind_dim.d[1], bind_dim.d[2], bind_dim.d[3]};
            if(context_->setInputShape(input_name_.c_str(), bind_dim)) {
                LOG_INFO("Input shape successfully created");
            } else {
                LOG_ERROR("Failed to set input shape");
            }
        } else if (io_mode == TensorIOMode::kOUTPUT) {
            output_name_ = tensor_name;
            LOG_INFO("Tensor RT output name: ", output_name_);
            Dims bind_dim = engine_->getTensorShape(tensor_name);
            output_shape_ = {bind_dim.d[0], bind_dim.d[1], bind_dim.d[2]};
        }
	}

    LOG_INFO(logger::log_seperator("Input Shape"));
	input_ = Tensor<float>(input_shape_, Device::CPU);
    LOG_INFO(input_.print_shape());
    LOG_INFO("CPU input addr: ", input_.data());

    input_.to_gpu();

    if(context_->setTensorAddress(input_name_.c_str(), input_.data())) {
        LOG_INFO("Set input tensor data successfully");
        LOG_INFO("GPU input addr: ", input_.data());
    } else {
        LOG_ERROR("Input setTensorAddress failed");
    }
    input_.to_cpu();

    LOG_INFO(logger::log_seperator("Output Shape"));
	output_ = Tensor<float>(output_shape_, Device::CPU);
    LOG_INFO(output_.print_shape());
    LOG_INFO("CPU output addr: ", output_.data());
    output_.to_gpu(); // make sure gpu buffer is created before hand
    if(context_->setTensorAddress(output_name_.c_str(), output_.data())) {
        LOG_INFO("Set output tensor data successfully");
        LOG_INFO("GPU output addr: ", output_.data());
    } else {
        LOG_ERROR("Output setTensorAddress failed");
    }

    output_.to_cpu(); // since TensorRT has the gpu ptr I can keep this on cpu and just force update cpu
                      
    if (stream_ == nullptr) {
        LOG_INFO("Cuda stream was not set externally, setting personal stream");
	    CUDA_CHECK(cudaStreamCreate(&stream_));
    }
//   
	//is_init_ = true;
}

void TensorRTEngine::run_inference() {
    auto start = std::chrono::high_resolution_clock::now();
    if (engine_ == nullptr || context_ == nullptr) {
        LOG_ERROR("Engine or Context not initialized");
    }
    input_.to_gpu();
    context_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);
    output_.update_cpu_from_gpu();
    //execute_cuda_graph();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli>elapsed = stop - start;
    LOG_INFO("Inference took: " + std::to_string(elapsed.count()));
}

Tensor<float>& TensorRTEngine::get_output_tensor() {
    return output_;
}

Tensor<float>& TensorRTEngine::get_input_tensor() {
    return input_;
}

const Shape TensorRTEngine::get_input_shape() const {
    return input_shape_; 
}
const Shape TensorRTEngine::get_output_shape() const {
    return output_shape_;
}

#endif
