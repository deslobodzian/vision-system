#include "preprocess_kernels.h"
#include "utils/logger.hpp"
// #include <opencv2/opencv.hpp> 


unsigned char* d_bgr = nullptr;
unsigned char* d_output = nullptr;

__global__ void kernel_preprocess_to_tensor(const unsigned char* d_bgr, float* d_output, int input_height, int input_width, int frame_s, int batch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < input_width && y < input_height) {
        int i = y * input_width + x;

        uchar3 pixel = make_uchar3(d_bgr[i * 3], d_bgr[i * 3 + 1], d_bgr[i * 3 + 2]);

        d_output[batch * 3 * frame_s + i] = (float)pixel.z / 255.0;
        d_output[batch * 3 * frame_s + i + frame_s] = (float)pixel.y / 255.0;
        d_output[batch * 3 * frame_s + i + 2 * frame_s] = (float)pixel.x / 255.0;
    }
}

__global__ void kernel_convert_to_bgr(float* input, unsigned char* output, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    const int inIdx = 4 * (y * width + x);  // BGRA input
    const int outIdx = 3 * (y * width + x); // BGR output
    
    if(inIdx + 3 >= width * height * 4 || outIdx + 2 >= width * height * 3) {
        printf("Thread (%d, %d) out of bounds: inIdx = %d, outIdx = %d\n", x, y, inIdx, outIdx);
        return;
    }

    output[outIdx]     = input[inIdx];
    output[outIdx + 1] = input[inIdx + 1];
    output[outIdx + 2] = input[inIdx + 2];
}

__global__ void kernel_convert_to_rgb(unsigned char* input, unsigned char* output, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    const int inIdx = 4 * (y * width + x);  // BGRA input
    const int outIdx = 3 * (y * width + x); // RGB output
    
    if(inIdx + 3 >= width * height * 4 || outIdx + 2 >= width * height * 3) {
        printf("Thread (%d, %d) out of bounds: inIdx = %d, outIdx = %d\n", x, y, inIdx, outIdx);
        return;
    }

    output[outIdx]     = input[inIdx + 2];  // Red
    output[outIdx + 1] = input[inIdx + 1];  // Green
    output[outIdx + 2] = input[inIdx];      // Blue
}

__global__ void kernel_preprocess_letterbox(const unsigned char* d_bgr, unsigned char* d_output_image, int input_width, int input_height, int image_width, int image_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float r_w = (float)input_width / image_width;
    float r_h = (float)input_height / image_height;

    int w, h, x_offset, y_offset;
    if (r_h > r_w) {
        w = input_width;
        h = r_w * image_height;
        x_offset = 0;
        y_offset = (input_height - h) / 2;
    } else {
        w = r_h * image_width;
        h = input_height;
        x_offset = (input_width - w) / 2;
        y_offset = 0;
    }

    if (x < input_width && y < input_height) {
        uchar3 pixel;
        
        if (x >= x_offset && x < (x_offset + w) && y >= y_offset && y < (y_offset + h)) {
            int x_resized = (int)((x - x_offset) * (image_width / (float)w));
            int y_resized = (int)((y - y_offset) * (image_height / (float)h));
            
            x_resized = min(x_resized, image_width - 1);
            y_resized = min(y_resized, image_height - 1);
            
            int i_resized = y_resized * image_width + x_resized;
            
            pixel = make_uchar3(d_bgr[i_resized * 3], d_bgr[i_resized * 3 + 1], d_bgr[i_resized * 3 + 2]);
        } else {
            pixel.x = 128;
            pixel.y = 128;
            pixel.z = 128;
        }

        int i = y * input_width + x;
        d_output_image[i * 3] = pixel.x;
        d_output_image[i * 3 + 1] = pixel.y;
        d_output_image[i * 3 + 2] = pixel.z;
    }
}

void init_preprocess_resources(int image_width, int image_height, int input_width, int input_height) {
    CUDA_CHECK(cudaMalloc(&d_bgr, image_width * image_height * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_output, input_width * input_height * 3 * sizeof(unsigned char)));
}

void preprocess_sl(const sl::Mat& left_img, Tensor<float>& d_input, int input_width, int input_height, size_t frame_s, int batch, cudaStream_t& stream) {
    int image_width = left_img.getWidth();
    int image_height = left_img.getHeight();

    if (d_input.device() != Device::GPU) {
        d_input.to_gpu();
    }

    if (d_bgr == nullptr || d_output == nullptr) {
        init_preprocess_resources(image_width, image_height, input_width, input_height);
    }
    cudaError_t err;

    dim3 block(16, 16);
    dim3 grid_input((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);
    dim3 grid_output((input_width + block.x - 1) / block.x, (input_height + block.y - 1) / block.y);
    
    //kernel_convert_to_bgr<<<grid_input, block, 0, stream>>>(left_img.getPtr<sl::uchar1>(sl::MEM::GPU), d_bgr, image_width, image_height);
    //err = cudaGetLastError();
    //if (err != cudaSuccess) {
    //    printf("kernel_convert_to_bgr launch failed: %s\n", cudaGetErrorString(err));
    //    return;
    //}

    kernel_preprocess_letterbox<<<grid_output, block, 0, stream>>>(d_bgr, d_output, input_width, input_height, image_width, image_height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel_preprocess_and_letterbox launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    kernel_preprocess_to_tensor<<<grid_output, block, 0, stream>>>(d_output, d_input.data(), input_height, input_width, frame_s, batch);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel_preprocess_to_tensor launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    unsigned char* h_letter = new unsigned char[input_width * input_height * 3];
    err = cudaMemcpy(h_letter, d_output, input_width * input_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy Device to Host failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cv::Mat kernel_out(input_height, input_width, CV_8UC3, h_letter);
    std::string filename = "kernel_letter_output.png";
    cv::imwrite(filename, kernel_out);
    delete[] h_letter;
}

void preprocess_cv(const cv::Mat& img, Tensor<float>& d_input, cudaStream_t& stream) {
    int image_width = img.cols;
    int image_height = img.rows;

    if (d_input.device() != Device::GPU) {
        d_input.to_gpu();
    }
    // BCWH
    int batch = d_input.shape()[0] - 1;
    int input_width = d_input.shape()[2];
    int input_height = d_input.shape()[2];
    size_t frame_s = d_input.size();

    if (d_bgr == nullptr || d_output == nullptr) {
        init_preprocess_resources(image_width, image_height, input_width, input_height);
    }
    cudaError_t err;
    size_t bytes = img.rows * img.cols * img.channels() * sizeof(unsigned char);
    cudaMemcpy(d_bgr, img.data, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid_input((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);
    dim3 grid_output((input_width + block.x - 1) / block.x, (input_height + block.y - 1) / block.y);
    
    //kernel_convert_to_bgr<<<grid_input, block, 0, stream>>>(d_input.data(), d_bgr, image_width, image_height);
    //err = cudaGetLastError();
    //if (err != cudaSuccess) {
    //    printf("kernel_convert_to_bgr launch failed: %s\n", cudaGetErrorString(err));
    //    return;
    //}

    kernel_preprocess_letterbox<<<grid_output, block, 0, stream>>>(d_bgr, d_output, input_width, input_height, image_width, image_height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel_preprocess_and_letterbox launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    kernel_preprocess_to_tensor<<<grid_output, block, 0, stream>>>(d_output, d_input.data(), input_height, input_width, frame_s, batch);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel_preprocess_to_tensor launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    //unsigned char* h_letter = new unsigned char[input_width * input_height * 3];
    //err = cudaMemcpy(h_letter, d_output, input_width * input_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) {
    //    LOG_ERROR("CUDA memcpy Device to Host failed: ", cudaGetErrorString(err));
    //    return;
    //}

    //cv::Mat kernel_out(input_height, input_width, CV_8UC3, h_letter);
    //std::string filename = "kernel_letter_output.png";
    //cv::imwrite(filename, kernel_out);
    //delete[] h_letter;
}
void free_preprocess_resources() {
    cudaFree(d_bgr);
    cudaFree(d_output);
}
