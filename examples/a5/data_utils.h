/**
 * @file data_utils.h
 * @brief Common functions used to read, write and print data.
 */
#pragma once

#include <acl/acl.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

enum class PrintDataType {
  DT_UNDEFINED = -1,
  FLOAT = 0,
  HALF = 1,
  INT8_T = 2,
  INT32_T = 3,
  UINT8_T = 4,
  INT16_T = 6,
  UINT16_T = 7,
  UINT32_T = 8,
  INT64_T = 9,
  UINT64_T = 10,
  DOUBLE = 11,
  BOOL = 12,
  STRING = 13,
  COMPLEX64 = 16,
  COMPLEX128 = 17,
  BF16 = 27
};

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_ACL(x)                                                    \
  do {                                                                  \
    const aclError __ret = x;                                           \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);

/**
 * @brief Read data from file.
 *
 * @param [in] file_path File path.
 * @param [out] buffer Pointer to the buffer where the data is read.
 * @param [in] size Size of the file and the buffer.
 * @return Boolean indicating if the data read was successful or not.
 */
bool ReadFile(const std::string& file_path, void* buffer, size_t size) {
  struct stat s_buf;
  const int file_status = stat(file_path.data(), &s_buf);
  if (file_status == -1) {
    ERROR_LOG("Failed to read file.");
    return false;
  }
  if (S_ISREG(s_buf.st_mode) == 0) {
    ERROR_LOG("File does not exist: %s", file_path.c_str());
    return false;
  }

  std::ifstream file;
  file.open(file_path, std::ios::binary);
  if (!file.is_open()) {
    ERROR_LOG("Failed to open file. Path = %s", file_path.c_str());
    return false;
  }

  std::filebuf* const buf = file.rdbuf();
  const size_t read_size = buf->pubseekoff(0, std::ios::end, std::ios::in);
  if (read_size == 0) {
    ERROR_LOG("%s: File is empty.", file_path.c_str());
    file.close();
    return false;
  }
  if (read_size > size) {
    ERROR_LOG("%s: File size is larger than the buffer size.",
              file_path.c_str());
    file.close();
    return false;
  }
  buf->pubseekpos(0, std::ios::in);
  buf->sgetn(static_cast<char*>(buffer), read_size);
  file.close();
  return true;
}

/**
 * @brief Write data to file.
 *
 * @param [in] file_path File path.
 * @param [in] buffer Data to write to file.
 * @param [in] size Size to write.
 * @return Boolean indicating if the data write was successful or not.
 */
bool WriteFile(const std::string& file_path, const void* buffer, size_t size) {
  if (buffer == nullptr) {
    ERROR_LOG("Cannot write file from a nullptr buffer.");
    return false;
  }

  const int fd =
      open(file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
  if (fd < 0) {
    ERROR_LOG("Failed to open file. Path = %s", file_path.c_str());
    return false;
  }

  const size_t writeSize = write(fd, buffer, size);
  (void)close(fd);
  if (writeSize != size) {
    ERROR_LOG("Failed to write file.");
    return false;
  }

  return true;
}

/// @private
template <typename T>
void DoPrintData(const T* data, size_t count, size_t elements_per_row) {
  assert(elements_per_row != 0);
  for (size_t i = 0; i < count; ++i) {
    if constexpr (std::is_same<T, int8_t>::value ||
                  std::is_same<T, uint8_t>::value) {
      // cout treats int8 as char and doesn't output its numeric
      // representation
      std::cout << std::setw(10) << static_cast<int>(data[i]);
    } else {
      std::cout << std::setw(10) << data[i];
    }
    if (i % elements_per_row == elements_per_row - 1) {
      std::cout << std::endl;
    }
  }
}

/// @private
void DoPrintHalfData(const aclFloat16* data, size_t count,
                     size_t elements_per_row) {
  assert(elements_per_row != 0);
  for (size_t i = 0; i < count; ++i) {
    std::cout << std::setw(10) << std::setprecision(6)
              << aclFloat16ToFloat(data[i]);
    if (i % elements_per_row == elements_per_row - 1) {
      std::cout << std::endl;
    }
  }
}

/**
 * @brief Print array content.
 *
 * @param [in] data Pointer to the array.
 * @param [in] count Number of elements to print.
 * @param [in] data_type Data type of the elements.
 * @param [in] elements_per_row Number of elements to be printed in a single
 * row.
 */
void PrintData(const void* data, size_t count, PrintDataType data_type,
               size_t elements_per_row = 16) {
  if (data == nullptr) {
    ERROR_LOG("Cannot print a nullptr buffer.");
    return;
  }

  switch (data_type) {
    case PrintDataType::BOOL:
      DoPrintData(reinterpret_cast<const bool*>(data), count, elements_per_row);
      break;
    case PrintDataType::INT8_T:
      DoPrintData(reinterpret_cast<const int8_t*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::UINT8_T:
      DoPrintData(reinterpret_cast<const uint8_t*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::INT16_T:
      DoPrintData(reinterpret_cast<const int16_t*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::UINT16_T:
      DoPrintData(reinterpret_cast<const uint16_t*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::INT32_T:
      DoPrintData(reinterpret_cast<const int32_t*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::UINT32_T:
      DoPrintData(reinterpret_cast<const uint32_t*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::INT64_T:
      DoPrintData(reinterpret_cast<const int64_t*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::UINT64_T:
      DoPrintData(reinterpret_cast<const uint64_t*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::HALF:
      DoPrintHalfData(reinterpret_cast<const aclFloat16*>(data), count,
                      elements_per_row);
      break;
    case PrintDataType::FLOAT:
      DoPrintData(reinterpret_cast<const float*>(data), count,
                  elements_per_row);
      break;
    case PrintDataType::DOUBLE:
      DoPrintData(reinterpret_cast<const double*>(data), count,
                  elements_per_row);
      break;
    default:
      ERROR_LOG("Unsupported type.");
  }
  std::cout << std::endl;
}

/**
 * @brief Prints beginning and end of a given vector.
 *
 * @param [in] data Pointer to the array.
 * @param [in] dt Data type of the elements.
 * @param [in] elems_to_print Number of elements to print both from the
 * beginning and end.
 * @param [in] vector_len Total number of elements in the vector.
 * @param [in] msg Additional message printed at the beginning.
 */
template <typename T>
void PrintVector(const T* data, PrintDataType dt, size_t elems_to_print,
                 size_t vector_len, std::string msg = "") {
  std::cout << "==========================================" << std::endl;
  if (msg != "") {
    std::cout << msg << std::endl;
  }
  if (2 * elems_to_print >= vector_len) {
    PrintData(data, vector_len, dt);
  } else {
    PrintData(data, elems_to_print, dt);
    std::cout << "\t..." << std::endl;
    const size_t tail_start = vector_len - elems_to_print;
    PrintData(data + tail_start, elems_to_print, dt);
  }
  std::cout << "==========================================" << std::endl;
}
