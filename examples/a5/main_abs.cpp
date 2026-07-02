/**
 *
 * @file main_abs.cpp
 * @brief Example of using the `abs` kernel.
 */

#include <acl/acl.h>

#include "data_utils.h"

extern "C" void call_vabs_fp16(uint32_t blockDim, aclrtStream stream, void* x,
                               void* z, uint32_t num_elements);

/// Number of elements in input vectors.
constexpr size_t VABS_TOTAL_LENGTH = 8 * 128;

int32_t main(int32_t argc, char* argv[]) {
  uint32_t blockDim;
  if (argc > 2) {
    std::cerr << "Usage: ./" << argv[0] << " <B=Number of blocks>" << std::endl;
    return 1;
  } else if (argc == 2) {
    blockDim = std::stoul(argv[1]);
    std::cout << "[vabs] Use input BlockDim: " << blockDim << std::endl;
  } else {
    std::cout << "[vabs] Use default BlockDim: 8" << std::endl;
    blockDim = 8;
  }

  constexpr size_t inputByteSize = VABS_TOTAL_LENGTH * sizeof(uint16_t);
  constexpr size_t outputByteSize = VABS_TOTAL_LENGTH * sizeof(uint16_t);

  CHECK_ACL(aclInit(nullptr));
  aclrtContext context;
  const int32_t device_id = 0;
  CHECK_ACL(aclrtSetDevice(device_id));
  CHECK_ACL(aclrtCreateContext(&context, device_id));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *xHost, *zHost;
  uint8_t *xDevice, *zDevice;
  CHECK_ACL(aclrtMallocHost((void**)&xHost, inputByteSize));
  CHECK_ACL(aclrtMallocHost((void**)&zHost, outputByteSize));
  CHECK_ACL(
      aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(
      aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input/input_x.bin", xHost, inputByteSize);
  PrintVector((uint16_t*)xHost, PrintDataType::HALF, 16, VABS_TOTAL_LENGTH,
              "Input X");
  CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  std::cout << "Init vabs_fp16 kernel" << std::endl;
  call_vabs_fp16(blockDim, stream, xDevice, zDevice, VABS_TOTAL_LENGTH);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  PrintVector((uint16_t*)zHost, PrintDataType::HALF, 16, VABS_TOTAL_LENGTH,
              "Output");
  WriteFile("vabs_output.bin", zHost, outputByteSize);

  CHECK_ACL(aclrtFree(xDevice));
  CHECK_ACL(aclrtFree(zDevice));
  CHECK_ACL(aclrtFreeHost(xHost));
  CHECK_ACL(aclrtFreeHost(zHost));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(device_id));
  CHECK_ACL(aclFinalize());
  return 0;
}
