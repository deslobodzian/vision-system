import re
from formatter import FileBase, FormatterBase
import subprocess

class CPPFormatter(FormatterBase):
    def __init__(self, root_dir):
        super().__init__(root_dir, [".cc", ".cpp", ".h", ".hpp"])

        self.cpp_files = [CPPFiles(file_path) for file_path in self.file_dirs]
        print(self.cpp_files)

    def format(self):
        """Format with clang-format"""
        success = True
        for cpp_file in self.cpp_files:
            if not cpp_file.run_clang_format():
                success = False
                print("Failed to format file: {self.cpp_file}")
        return success

"""
Handles '.cc', '.cpp', '.h', '.hpp' file extensions
"""
class CPPFiles(FileBase):
    def __init__(self, file_path):
        super().__init__(file_path)

    def includes(self):
        regex = r'#include\s*[<"]([^">]*)[>"]'
        # walrus operator is sick
        out = [match[0] for line in self.lines if (match := re.findall(regex, line))]
        return out

    def define_values(self):
        regex = r'#define\s+(\w+)\s([^\n(].*)'
        out = list(match[0] for line in self.lines if (match := re.findall(regex, line)))
        return out

    def __repr__(self):
        return f"{self.name}{self.extension}"

    def run_clang_format(self, style="file"):
        cmd = ["clang-format", "-i"]
        if style:
            cmd.extend(["-style=" + style])
        cmd.append(str(self.file_path))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            with open(self.file_path, "r") as file:
                self.lines = file.readlines()
            return True
        except subprocess.CalledProccessError as e:
            print(f"Failed to format: {self.file_path}: {e.stderr}")
            return False


if __name__ == "__main__":
    cpp_formatter = CPPFormatter("../../common")
    success = cpp_formatter.format()

    if success:
        print("Files formatted")

