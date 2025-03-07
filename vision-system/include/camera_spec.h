#pragma once
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <cmath>  // For std::abs

namespace camera {

struct Resolution {
    int width_;
    int height_;

    Resolution() : width_(0), height_(0) {}
    Resolution(int w, int h) : width_(w), height_(h) {}

    bool operator==(const Resolution& other) const {
        return width_ == other.width_ && height_ == other.height_;
    }

    bool operator!=(const Resolution& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        return std::to_string(width_) + "x" + std::to_string(height_);
    }

    static Resolution hd() { return Resolution(1280, 720); }
    static Resolution full_hd() { return Resolution(1920, 1080); }
    static Resolution uhd() { return Resolution(3840, 2160); }
    static Resolution vga() { return Resolution(640, 480); }
};

enum class CameraBackend {
    OPENCV,
    ZED,
    V4L2,
    FFMPEG_BACKEND,
    UNKNOWN
};

inline std::string backend_to_string(CameraBackend backend) {
    switch (backend) {
        case CameraBackend::OPENCV: return "OpenCV";
        case CameraBackend::ZED: return "ZED";
        case CameraBackend::V4L2: return "V4L2";
        case CameraBackend::FFMPEG_BACKEND: return "FFmpeg";
        default: return "Unknown";
    }
}

inline CameraBackend backend_from_string(const std::string& backend_str) {
    if (backend_str == "OpenCV") return CameraBackend::OPENCV;
    if (backend_str == "ZED") return CameraBackend::ZED;
    if (backend_str == "V4L2") return CameraBackend::V4L2;
    if (backend_str == "FFmpeg") return CameraBackend::FFMPEG_BACKEND;
    return CameraBackend::UNKNOWN;
}

enum class ImageFormat {
    RGB,
    BGR,
    RGBA,
    BGRA,
    GRAY,
    YUV,
    DEPTH,
    YUYV,   // Common FFmpeg format (YUV 4:2:2)
    NV12,   // Common FFmpeg format (YUV 4:2:0 semi-planar)
    MJPEG,  // Common FFmpeg format (Motion JPEG)
    UNKNOWN
};

inline std::string format_to_string(ImageFormat format) {
    switch (format) {
        case ImageFormat::RGB: return "RGB";
        case ImageFormat::BGR: return "BGR";
        case ImageFormat::RGBA: return "RGBA";
        case ImageFormat::BGRA: return "BGRA";
        case ImageFormat::GRAY: return "GRAY";
        case ImageFormat::YUV: return "YUV";
        case ImageFormat::DEPTH: return "DEPTH";
        case ImageFormat::YUYV: return "YUYV";
        case ImageFormat::NV12: return "NV12";
        case ImageFormat::MJPEG: return "MJPEG";
        default: return "Unknown";
    }
}

inline ImageFormat format_from_string(const std::string& format_str) {
    if (format_str == "RGB") return ImageFormat::RGB;
    if (format_str == "BGR") return ImageFormat::BGR;
    if (format_str == "RGBA") return ImageFormat::RGBA;
    if (format_str == "BGRA") return ImageFormat::BGRA;
    if (format_str == "GRAY") return ImageFormat::GRAY;
    if (format_str == "YUV") return ImageFormat::YUV;
    if (format_str == "DEPTH") return ImageFormat::DEPTH;
    if (format_str == "YUYV") return ImageFormat::YUYV;
    if (format_str == "NV12") return ImageFormat::NV12;
    if (format_str == "MJPEG") return ImageFormat::MJPEG;
    return ImageFormat::UNKNOWN;
}

struct CameraIntrinsics {
    double fx_;       // Focal length x
    double fy_;       // Focal length y
    double cx_;       // Principal point x
    double cy_;       // Principal point y
    std::vector<double> distortion_coeffs_; // Distortion coefficients
    bool is_calibrated_;

    CameraIntrinsics() : fx_(0), fy_(0), cx_(0), cy_(0), is_calibrated_(false) {}
};

struct FieldOfView {
    double horizontal_;   // Horizontal FOV in degrees
    double vertical_;     // Vertical FOV in degrees
    double diagonal_;     // Diagonal FOV in degrees

    FieldOfView() : horizontal_(0), vertical_(0), diagonal_(0) {}
};

struct CameraSpec {
    int id_;                              // Camera ID or index
    std::string name_;                    // Camera name/model
    std::string device_path_;             // Device path (for FFmpeg/V4L2)
    CameraBackend backend_;               // Camera backend type

    std::vector<Resolution> supported_resolutions_;  // Available resolutions
    std::vector<double> supported_fps_;              // Available frame rates
    std::vector<ImageFormat> supported_formats_;     // Supported image formats

    Resolution current_resolution_;        // Current resolution
    double current_fps_;                   // Current frame rate
    ImageFormat current_format_;           // Current format

    std::optional<CameraIntrinsics> intrinsics_;    // Optional calibration data
    std::optional<FieldOfView> fov_;                // Optional field of view

    bool has_depth_capability_;            // Depth sensing capability
    bool has_ir_capability_;               // Infrared capability
    bool is_stereo_cam_;                   // Is a stereo camera

    // Additional camera capabilities/features as key-value pairs
    std::unordered_map<std::string, std::string> additional_capabilities_;

    CameraSpec() :
        id_(-1),
        backend_(CameraBackend::UNKNOWN),
        current_fps_(0),
        current_format_(ImageFormat::UNKNOWN),
        has_depth_capability_(false),
        has_ir_capability_(false),
        is_stereo_cam_(false) {}

    bool supports_resolution(const Resolution& res) const {
        for (const auto& supported_res : supported_resolutions_) {
            if (supported_res == res) return true;
        }
        return false;
    }

    bool supports_fps(double fps) const {
        for (const auto& rate : supported_fps_) {
            if (std::abs(rate - fps) < 0.001) return true;
        }
        return false;
    }

    bool supports_format(ImageFormat format) const {
        for (const auto& fmt : supported_formats_) {
            if (fmt == format) return true;
        }
        return false;
    }

    bool has_supported_resolutions() const {
        return !supported_resolutions_.empty();
    }

    bool has_supported_framerates() const {
        return !supported_fps_.empty();
    }

    bool has_supported_formats() const {
        return !supported_formats_.empty();
    }

    std::string to_string() const {
        std::string result = "Camera " + std::to_string(id_) + " (" + name_ + ")\n";
        if (!device_path_.empty()) {
            result += "  Device path: " + device_path_ + "\n";
        }
        result += "  Backend: " + backend_to_string(backend_) + "\n";
        result += "  Current resolution: " + current_resolution_.to_string() + "\n";
        result += "  Current FPS: " + std::to_string(current_fps_) + "\n";
        result += "  Current format: " + format_to_string(current_format_) + "\n";

        if (!supported_resolutions_.empty()) {
            result += "  Supported resolutions: ";
            for (size_t i = 0; i < supported_resolutions_.size(); ++i) {
                if (i > 0) result += ", ";
                result += supported_resolutions_[i].to_string();
            }
            result += "\n";
        }

        if (!supported_fps_.empty()) {
            result += "  Supported FPS: ";
            for (size_t i = 0; i < supported_fps_.size(); ++i) {
                if (i > 0) result += ", ";
                result += std::to_string(supported_fps_[i]);
            }
            result += "\n";
        }

        result += "  Depth capability: " + std::string(has_depth_capability_ ? "Yes" : "No") + "\n";
        result += "  IR capability: " + std::string(has_ir_capability_ ? "Yes" : "No") + "\n";
        result += "  Stereo camera: " + std::string(is_stereo_cam_ ? "Yes" : "No") + "\n";

        return result;
    }
};

} // namespace camera
