import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile

from pathlib import Path


# class DependencyInstaller():
#    def __init__(self):
#        self.platform = platform.system()

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

def install_opencv():
    version = "4.10.0"
    system = platform.system().lower()
    cwd = Path.cwd()
    
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

    install_dir = "C:/Program Files/OpenCV" if system == "windows" else "/usr/local"

    if system == "windows":
        cmake_config = [
            "cmake", "..",
            "-A", "x64",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DOPENCV_EXTRA_MODULES_PATH={contrib_dir}/modules",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DCMAKE_CONFIGURATION_TYPES=Release",
            "-DINSTALL_PYTHON_EXAMPLES=OFF",
            "-DINSTALL_C_EXAMPLES=OFF",
            "-DOPENCV_ENABLE_NONFREE=ON",
            "-DOPENCV_GENERATE_PKGCONFIG=ON"
        ]
    else:
        cmake_config = [
            "cmake", "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=/usr/local",
            f"-DOPENCV_EXTRA_MODULES_PATH={contrib_dir}/modules",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DINSTALL_PYTHON_EXAMPLES=OFF",
            "-DINSTALL_C_EXAMPLES=OFF",
            "-DOPENCV_ENABLE_NONFREE=ON",
            "-DOPENCV_GENERATE_PKGCONFIG=ON"
        ]

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
        opencv_bin = Path(install_dir) / 'x64' / 'vc17' / 'bin'
        opencv_config = Path(install_dir) / 'x64' / 'vc17' / 'lib'
        
        os.environ['PATH'] += os.pathsep + str(opencv_bin)
        subprocess.run(["setx", "PATH", f"%PATH%;{opencv_bin}"], capture_output=True)
        
        subprocess.run(["setx", "OpenCV_DIR", str(opencv_config)], capture_output=True)
        
        print(f"\nOpenCV installed at: {install_dir}")
        print(f"OpenCV binaries added to PATH: {opencv_bin}")
        print(f"Set OpenCV_DIR for CMake to: {opencv_config}")
    else:
        subprocess.run(["ldconfig"])
        print("\nOpenCV installed to /usr/local")
        print("CMake should find OpenCV automatically")


if __name__ == "__main__":
    install_opencv()