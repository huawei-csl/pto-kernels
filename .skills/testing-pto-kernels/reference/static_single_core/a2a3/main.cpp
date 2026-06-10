#include <acl/acl.h>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <vector>

using AddFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);

namespace {
constexpr size_t kElems = 64 * 64;
constexpr size_t kBytes = kElems * sizeof(uint16_t);
constexpr uint16_t kHalfOne = 0x3c00;
constexpr uint16_t kHalfTwo = 0x4000;
constexpr uint16_t kHalfThree = 0x4200;

void check_acl(aclError ret, const char *what) {
  if (ret != ACL_SUCCESS) {
    std::cerr << what << " failed, aclError=" << ret << "\n";
    std::exit(1);
  }
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " /path/to/libstatic_add_a2a3.so\n";
    return 2;
  }
  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::cerr << "dlopen failed: " << dlerror() << "\n";
    return 1;
  }
  auto call_add = reinterpret_cast<AddFn>(dlsym(handle, "call_static_add"));
  if (call_add == nullptr) {
    std::cerr << "dlsym failed: call_static_add\n";
    return 1;
  }

  std::vector<uint16_t> x(kElems, kHalfOne), z(kElems, kHalfTwo), out(kElems, 0);
  int device_id = 0;
  if (const char *env = std::getenv("NPU_DEVICE")) {
    std::string value(env);
    auto pos = value.find(':');
    device_id = std::stoi(pos == std::string::npos ? value : value.substr(pos + 1));
  }

  check_acl(aclInit(nullptr), "aclInit");
  check_acl(aclrtSetDevice(device_id), "aclrtSetDevice");
  aclrtStream stream = nullptr;
  check_acl(aclrtCreateStream(&stream), "aclrtCreateStream");

  void *x_dev = nullptr, *z_dev = nullptr, *out_dev = nullptr;
  check_acl(aclrtMalloc(&x_dev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
  check_acl(aclrtMalloc(&z_dev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc z");
  check_acl(aclrtMalloc(&out_dev, kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc out");
  check_acl(aclrtMemcpy(x_dev, kBytes, x.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy x");
  check_acl(aclrtMemcpy(z_dev, kBytes, z.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy z");
  call_add(1, stream, static_cast<uint8_t *>(out_dev), static_cast<uint8_t *>(x_dev),
           static_cast<uint8_t *>(z_dev));
  check_acl(aclrtSynchronizeStream(stream), "sync");
  check_acl(aclrtMemcpy(out.data(), kBytes, out_dev, kBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy out");

  size_t errors = 0;
  for (uint16_t v : out) errors += (v != kHalfThree);

  aclrtFree(out_dev);
  aclrtFree(z_dev);
  aclrtFree(x_dev);
  aclrtDestroyStream(stream);
  aclrtResetDevice(device_id);
  aclFinalize();

  if (errors) {
    std::cerr << "FAIL static A2A3 ACL add errors=" << errors << "\n";
    return 1;
  }
  std::cout << "PASS static_single_core/a2a3 ACL add\n";
  return 0;
}
