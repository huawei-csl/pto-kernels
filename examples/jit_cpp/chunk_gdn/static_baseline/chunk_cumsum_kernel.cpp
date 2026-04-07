#include "common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ float *G_handle, __gm__ float *S_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  chunk_gdn_pto::TileUbDataND<float, 1, 1024, 1, 1024> s_ub;
  TASSIGN(s_ub, 0);
  chunk_gdn_pto::TileUbDataND<float, 1, 1024, 1, 1024> g_ub;
  TASSIGN(g_ub, 4096);
  auto vid = get_subblockid();
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(s_ub, 0.000000e+00f);
    chunk_gdn_pto::copy_gm_to_ub<float, float, 1, 1, 1, 1, 1024, 1, 1, 1, 524288, 1, 1, 1024, pto::PadValue::Zero>(G_handle + ((((cid / 16) * 32768) + (vid * 16384)) + ((cid % 16) * 1024)), 4096, 0, 1, 1024);
    chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);

  for (int32_t ii = 0; ii < 8; ++ii) {
      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_S> (0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_S> (0);
      s_ub.SetValue((ii * 128), g_ub.GetValue((ii * 128)));

  for (int32_t i = 1; i < 128; ++i) {
        float tmp2 = (s_ub.GetValue((((ii * 128) + i) - 1)) + g_ub.GetValue(((ii * 128) + i)));
        s_ub.SetValue(((ii * 128) + i), tmp2);
      }
      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
      chunk_gdn_pto::copy_ub_to_gm<float, float, 1, 1, 1, 1, 1024, 1, 1, 1, 524288, 1, 1, 1024>(S_handle + ((((cid / 16) * 32768) + (vid * 16384)) + ((cid % 16) * 1024)), 0, 0, 1, 1024);
    }
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *G_handle, __gm__ uint8_t *S_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ float *>(G_handle),
     reinterpret_cast<__gm__ float *>(S_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *G_handle, uint8_t *S_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<256, nullptr, stream>>>(G_handle, S_handle, fftsAddr);
}
