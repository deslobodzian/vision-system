import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
import argparse
from pathlib import Path


def is_opencv_installed(install_dir=None):
    """Check if OpenCV is already installed on the system or in the specified directory."""
    system = platform.system().lower()

    # First check the specified install directory if provided
    if install_dir and os.path.exists(install_dir):
        opencv_cmake_dir = Path(install_dir) / "lib" / "cmake" / "opencv4"
        if opencv_cmake_dir.exists():
            print(f"OpenCV installation found at specified directory: {install_dir}")
            return True

    if system == "windows":
        opencv_paths = [
            "C:/Program Files/OpenCV",
            "C:/OpenCV",
            os.environ.get("OpenCV_DIR", "")
        ]

        for path in opencv_paths:
            if path and os.path.exists(path):
                print(f"OpenCV installation found at: {path}")
                return True

        if "OpenCV_DIR" in os.environ:
            print(f"OpenCV_DIR environment variable found: {os.environ['OpenCV_DIR']}")
            return True

    elif system == "linux":
        result = subprocess.run(
            ["pkg-config", "--exists", "opencv4"],
            capture_output=True,
            check=False
        )
        if result.returncode == 0:
            version = subprocess.check_output(
                ["pkg-config", "--modversion", "opencv4"],
                universal_newlines=True
            ).strip()
            print(f"OpenCV is already installed (pkg-config). Version: {version}")
            return True

        # Check common system paths
        opencv_paths = [
            "/usr/include/opencv4",
            "/usr/local/include/opencv4"
        ]

        for path in opencv_paths:
            if os.path.exists(path):
                print(f"OpenCV headers found at: {path}")
                return True

    elif system == "darwin":  # macOS
        result = subprocess.run(
            ["brew", "list", "opencv"],
            capture_output=True,
            check=False
        )
        if result.returncode == 0:
            print("OpenCV is already installed via Homebrew")
            return True

        opencv_paths = [
            "/usr/local/include/opencv4",
            "/opt/homebrew/include/opencv4"
        ]

        for path in opencv_paths:
            if os.path.exists(path):
                print(f"OpenCV headers found at: {path}")
                return True

    print("OpenCV is not installed or not found in common locations")
    return False

def download_file(url, filename):
    print(f"Download {filename}...")
    urllib.request.urlretrieve(url, filename)

def extract_zip(filename, extract_path=""):
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filename, "r") as zip_file:
        zip_file.extractall(extract_path)

# Windows only
def install_choco():
    if platform.system().lower() != "windows":
        print("System is not windows, not installing Chocolately")
        return False

    if shutil.which("choco"):
        print("Chocolatey already installed for Windows")
        return False

    powershell_command = (
        "Set-ExecutionPolicy Bypass -Scope Process -Force; "
        "[System.Net.ServicePointManager]::SecurityProtocol = "
        "[System.Net.ServicePointManager]::SecurityProtocol -bor 3072; "
        "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    )

    try:
        subprocess.run(
            ["powershell", "-Command", powershell_command],
            check=True,
            shell=True
        )
        print("Chocolatey installation command executed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False

def install_cmake():
    if shutil.which("cmake"):
        print("CMake already installed")
        # return False

    system = platform.system().lower()
    command = 'echo "NULL COMMAND"'
    if system == "windows":
        command = ["powershell", "-Command", ("choco install cmake -y")]

    if system == "linux":
        command = ["bash", ("sudo apt install cmake")]

    print(command)
    try:
        subprocess.run(
            command,
            check=True,
            shell=True
        )
        print("CMake installation command executed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error occured: {e}")
        return False

def install_opencv(install_dir=None):
    version = "4.10.0"
    system = platform.system().lower()
    cwd = Path.cwd()

    if not install_dir:
        install_dir = "C:/Program Files/OpenCV" if system == "windows" else "/usr/local"
    else:
        os.makedirs(install_dir, exist_ok=True)

    if is_opencv_installed(install_dir):
        return

    opencv_url = f"https://github.com/opencv/opencv/archive/{version}.zip"
    contrib_url = f"https://github.com/opencv/opencv_contrib/archive/{version}.zip"

    download_file(opencv_url, "opencv.zip")
    download_file(contrib_url, "opencv_contrib.zip")

    extract_zip("opencv.zip")
    extract_zip("opencv_contrib.zip")

    opencv_dir = cwd / f"opencv-{version}"
    contrib_dir = cwd / f"opencv_contrib-{version}"
    build_dir = opencv_dir / "build"
    build_dir.mkdir(exist_ok=True)
    os.chdir(build_dir)

    cmake_config = [
        "cmake", "..",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DOPENCV_EXTRA_MODULES_PATH={contrib_dir}/modules",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DINSTALL_PYTHON_EXAMPLES=OFF",
        "-DINSTALL_C_EXAMPLES=OFF",
        "-DOPENCV_ENABLE_NONFREE=ON",
        "-DOPENCV_GENERATE_PKGCONFIG=ON"
    ]

    if system == "windows":
        cmake_config.extend(["-A", "x64", "-DCMAKE_CONFIGURATION_TYPES=Release"])

    print("Configuring with CMake...")
    process = subprocess.Popen(
        cmake_config,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()

    print("\nBuilding OpenCV...")
    build_command = ["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count())]
    process = subprocess.Popen(
        build_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()

    print("\nInstalling OpenCV...")
    install_command = ["cmake", "--install", "."]
    process = subprocess.Popen(
        install_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()

    if system == "windows":
        if install_dir.lower() in ["c:/program files/opencv", "c:/opencv"]:
            opencv_bin = Path(install_dir) / 'x64' / 'vc17' / 'bin'
            opencv_config = Path(install_dir) / 'x64' / 'vc17' / 'lib'

            os.environ['PATH'] += os.pathsep + str(opencv_bin)
            subprocess.run(["setx", "PATH", f"%PATH%;{opencv_bin}"], capture_output=True)
            subprocess.run(["setx", "OpenCV_DIR", str(opencv_config)], capture_output=True)

        print(f"\nOpenCV installed at: {install_dir}")
    elif system == "darwin":
        if install_dir == "/usr/local":
            subprocess.run(["update_dyld_shared_cache"])
        print(f"\nOpenCV installed to {install_dir}")
    else:
        if install_dir == "/usr/local":
            subprocess.run(["ldconfig"])
        print(f"\nOpenCV installed to {install_dir}")

    print(f"For CMake, use -DOpenCV_DIR={install_dir}/lib/cmake/opencv4")

    if os.path.exists("opencv.zip"):
        os.remove("opencv.zip")
    if os.path.exists("opencv_contrib.zip"):
        os.remove("opencv_contrib.zip")

    os.chdir(cwd)

    if os.path.exists(opencv_dir):
        shutil.rmtree(opencv_dir)
    if os.path.exists(contrib_dir):
        shutil.rmtree(contrib_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Install OpenCV with customizable install directory')
    parser.add_argument('--install-dir', type=str, help='Directory to install OpenCV')

    args = parser.parse_args()
    install_opencv(args.install_dir)
