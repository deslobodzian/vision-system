import os
from abc import ABC, abstractmethod
from pathlib import Path

def find_files(root_dir, extensions):
    root = Path(root_dir)

    return (file for ext in extensions for file in root.rglob(f"*{ext}"))

"""
    Base class for formatters for this project
"""
class FormatterBase(ABC):
    def __init__(self, root_dir=None, ext_list=None):
        self.root_dir = os.path.abspath(root_dir)
        self.ext_list = ext_list
        print(self.root_dir)

        self.file_dirs = []
        for filepath in find_files(self.root_dir, self.ext_list):
            self.file_dirs.append(filepath)

    @abstractmethod
    def format(self):
        pass


class FileBase:
    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, "r") as self.file:
            self.lines = self.file.readlines()

        print(f"File: {os.path.basename(file_path)}, has line count: {self.line_count()}")
        self.name = Path(file_path).stem
        self.extension = Path(file_path).suffix

    def line_count(self):
        return len(self.lines)

    def character_count(self):
        return sum(len(line) for line in self.lines)

    def absolute_path(self):
        return os.path.abspath(self.file_path)

