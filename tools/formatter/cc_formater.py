from formatter import FileBase
import re

class CCFile(FileBase):
    def __init__(self, file_path):
        super().__init__(file_path)


    def includes(self):
        regex = r"#include"
        out = [re.findall(regex, line) for line in self.lines]
        out = list(filter(None, out))
        print(out)


if __name__ == "__main__":
    file = CCFile("/Users/deslobodzian/Documents/Projects/vision-system/common/logger/logger.h")
    file.includes()

