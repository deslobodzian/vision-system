#ifndef VISION_SYSTEM_UTILS_HPP
#define VISION_SYSTEM_UTILS_HPP
#include <fstream>
#include <string>
#include <vector>

inline std::vector<std::string> split_str(std::string s,
                                          std::string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

inline bool readFile(std::string filename, std::vector<uint8_t> &file_content) {
  // open the file:
  std::ifstream instream(filename, std::ios::in | std::ios::binary);
  if (!instream.is_open())
    return true;
  file_content =
      std::vector<uint8_t>((std::istreambuf_iterator<char>(instream)),
                           std::istreambuf_iterator<char>());
  return false;
}

inline std::string remove_file_extension(const std::string &file) {
  size_t last_period = file.find_last_of('.');
  if (last_period != std::string::npos) {
    return file.substr(0, last_period);
  }
  return file;
}

#endif /* VISION_SYSTEM_UTILS_HPP */
