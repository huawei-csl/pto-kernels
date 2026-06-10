#include <acl/acl.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

extern "C" void call_static_add(uint32_t block_dim, void *stream, uint8_t *out,
                                uint8_t *x, uint8_t *z);

namespace {
constexpr size_t kElems = 64 * 64;
constexpr size_t kBytes = kElems * sizeof(uint16_t);
constexpr uint16_t kHalfOne = 0x3c00;    // fp16 1.0
constexpr uint16_t kHalfTwo = 0x4000;    // fp16 2.0
constexpr uint16_t kHalfThree = 0x4200;  // fp16 3.0

void check_acl(aclError ret, const char *what) {
  if (ret != ACL_SUCCESS) {
    std::cerr << what << " failed, aclError=" << ret << "\n";
    std::exit(1);
  }
}
}  // namespace

int main() {
  std::vector<uint16_t> x(kElems, kHalfOne);
  std::vector<uint16_t> z(kElems, kHalfTwo);
  std::vector<uint16_t> out(kElems, 0);

  check_acl(aclInit(nullptr), "aclInit");
  check_acl(aclrtSetDevice(0), "aclrtSetDevice");
  aclrtStream stream = nullptr;
  check_acl(aclrtCreateStream(&stream), "aclrtCreateStream");

  void *x_dev = nullptr;
  void *z_dev = nullptr;
  void *out_dev = nullptr;
  check_acl(aclrtMalloc(&x_dev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc x");
  check_acl(aclrtMalloc(&z_dev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc z");
  check_acl(aclrtMalloc(&out_dev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc out");
  check_acl(aclrtMemcpy(x_dev, kBytes, x.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy x");
  check_acl(aclrtMemcpy(z_dev, kBytes, z.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy z");

  call_static_add(1, stream, static_cast<uint8_t *>(out_dev), static_cast<uint8_t *>(x_dev),
                  static_cast<uint8_t *>(z_dev));
  check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
  check_acl(aclrtMemcpy(out.data(), kBytes, out_dev, kBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy out");

  size_t errors = 0;
  for (uint16_t v : out) {
    if (v != kHalfThree) ++errors;
  }

  aclrtFree(out_dev);
  aclrtFree(z_dev);
  aclrtFree(x_dev);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  if (errors != 0) {
    std::cerr << "FAIL static A5 ACL/runtime_camodel add errors=" << errors << "\n";
    return 1;
  }
  std::cout << "PASS static_single_core/a5 ACL runtime_camodel add\n";
  return 0;
}
