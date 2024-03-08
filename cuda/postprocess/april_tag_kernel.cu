#include "april_tag_kernel.h"
#include "sl/Camera.hpp"
#include <cuda_runtime_api.h>
#include "utils/logger.hpp"

__device__ bool is_valid_depth(float depth) {
    return depth > 0.0f && depth < 1000.0f; // Adjust the range according to your requirements
}

__device__ sl::float3 normalize(const sl::float3& v) {
    float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0.0f) {
        return sl::float3(v.x / length, v.y / length, v.z / length);
    }
    return v;
}

__device__ void normalize_device_orientation(DeviceOrientation& orientation) {
    float norm = sqrt(orientation.data[0] * orientation.data[0] +
                      orientation.data[1] * orientation.data[1] +
                      orientation.data[2] * orientation.data[2] +
                      orientation.data[3] * orientation.data[3]);
    for (int i = 0; i < 4; ++i) {
        orientation.data[i] /= norm;
    }
}

__device__ DeviceOrientation compute_orientation_from_normal(const sl::float3& normal) {
    sl::float3 up_vector = {0.0f, 0.0f, 1.0f};
    sl::float3 normalized_normal = normalize(normal);

    sl::float3 rotation_axis = {
        up_vector.y * normalized_normal.z - up_vector.z * normalized_normal.y,
        up_vector.z * normalized_normal.x - up_vector.x * normalized_normal.z,
        up_vector.x * normalized_normal.y - up_vector.y * normalized_normal.x
    };

    rotation_axis = normalize(rotation_axis);

    float dot_product = sl::float3::dot(up_vector, normalized_normal);
    float angle = acos(dot_product);

    float s = sin(angle / 2);

    DeviceOrientation orientation;
    orientation.data[0] = rotation_axis.x * s;
    orientation.data[1] = rotation_axis.y * s;
    orientation.data[2] = rotation_axis.z * s;
    orientation.data[3] = cos(angle / 2);

    normalize_device_orientation(orientation);

    return orientation;
}

__global__ void calculate_zed_apriltag_kernel(const sl::uchar4* point_cloud, size_t point_cloud_step, const sl::uchar4* normals, size_t normals_step,
                                              const cuAprilTagsID_t* detections, ZedAprilTag* zed_tags, int num_detections) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_detections) {
        const cuAprilTagsID_t& tag = detections[tid];
        ZedAprilTag z_tag;

        sl::float3 average_normal = {0, 0, 0};
        int valid_corners = 0;

        for (int i = 0; i < 4; ++i) {
            size_t point_offset = tag.corners[i].y * point_cloud_step + tag.corners[i].x * sizeof(sl::uchar4);
            if (point_offset >= point_cloud_step * tag.corners[i].y) {
                printf("Invalid point_offset: %zu\n", point_offset);
                return;
            }
            const sl::uchar4* point_ptr = reinterpret_cast<const sl::uchar4*>(reinterpret_cast<const unsigned char*>(point_cloud) + point_offset);
            sl::float4 point = sl::float4(point_ptr->x, point_ptr->y, point_ptr->z, point_ptr->w);

            if (is_valid_depth(point.z)) {
                z_tag.corners[valid_corners] = point;
                z_tag.center += point;
                valid_corners++;

                 size_t normal_offset = tag.corners[i].y * normals_step + tag.corners[i].x * sizeof(sl::uchar4);
                 const sl::uchar4* normal_ptr = reinterpret_cast<const sl::uchar4*>(reinterpret_cast<const unsigned char*>(normals) + normal_offset);
                 sl::float4 corner_normal = sl::float4(normal_ptr->x, normal_ptr->y, normal_ptr->z, normal_ptr->w);

                average_normal += sl::float3(corner_normal.x, corner_normal.y, corner_normal.z);
            }
        }

        if (valid_corners > 0) {
            z_tag.center /= valid_corners;
            average_normal /= valid_corners;

            z_tag.orientation = compute_orientation_from_normal(average_normal);
            z_tag.tag_id = tag.id;

            zed_tags[tid] = z_tag;
        } else {
            z_tag.tag_id = 9999999;
            zed_tags[tid] = z_tag;
        }
    }
}


std::vector<ZedAprilTag> detect_and_calculate(const sl::Mat& point_cloud, const sl::Mat& normals, const std::vector<cuAprilTagsID_t>& detections,
                                              cuAprilTagsID_t* gpu_detections, ZedAprilTag* gpu_zed_tags, int max_detections, cudaStream_t& stream) {
    int num_detections = detections.size();
    if (num_detections == 0) {
        return std::vector<ZedAprilTag>();
    }

    cudaError_t err = cudaMemcpyAsync(gpu_detections, detections.data(), num_detections * sizeof(cuAprilTagsID_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy detections to GPU: " + std::string(cudaGetErrorString(err)));
    }

    const sl::uchar4* gpu_point_cloud = point_cloud.getPtr<sl::uchar4>(sl::MEM::GPU);
    size_t point_cloud_step = point_cloud.getStepBytes(sl::MEM::GPU);
    const sl::uchar4* gpu_normals = normals.getPtr<sl::uchar4>(sl::MEM::GPU);
    size_t normals_step = normals.getStepBytes(sl::MEM::GPU);

    int block_size = 256;
    int num_blocks = (num_detections + block_size - 1) / block_size;
    calculate_zed_apriltag_kernel<<<num_blocks, block_size, 0, stream>>>(gpu_point_cloud, point_cloud_step, gpu_normals, normals_step,
                                                              gpu_detections, gpu_zed_tags, num_detections);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    std::vector<ZedAprilTag> zed_tags(num_detections);
    err = cudaMemcpyAsync(zed_tags.data(), gpu_zed_tags, num_detections * sizeof(ZedAprilTag), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy ZedAprilTags from GPU to CPU: " + std::string(cudaGetErrorString(err)));
    }

    return zed_tags;
}

