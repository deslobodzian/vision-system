import os
from pathlib import Path

"""
    Base class for formatters for this project
"""
class FormatterBase:
    def __init__():
        file_dirs = []


class FileBase:
    def __init__(self, file_path: str):
        with open(file_path, "r") as self.file:
            self.lines = self.file.readlines()

        print(f"File: {os.path.basename(file_path)}, has line count: {len(self.lines)}")
        print(self.lines[0])
        # for line in self.lines:
        #     print(line, end='')

    def line_count(self):
        return len(self.lines)

    def character_count(self):
        return sum(len(line) for line in self.lines)
