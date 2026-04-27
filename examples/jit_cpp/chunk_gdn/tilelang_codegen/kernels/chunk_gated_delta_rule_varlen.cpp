#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR h_handle,  GM_ADDR k_handle,  GM_ADDR v_handle,  GM_ADDR w_handle,  GM_ADDR g_handle,  GM_ADDR v_new_handle,  GM_ADDR h0_handle,  GM_ADDR ht_handle,  GM_ADDR cu_seqlens_handle,  GM_ADDR ws_wh_handle,  GM_ADDR ws_vnew_handle,  GM_ADDR ws_hupd_handle,  GM_ADDR ws_h_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> h;
  h.SetGlobalBuffer((__gm__ half*)h_handle);
  AscendC::GlobalTensor<half> k;
  k.SetGlobalBuffer((__gm__ half*)k_handle);
  AscendC::GlobalTensor<half> v;
  v.SetGlobalBuffer((__gm__ half*)v_handle);
  AscendC::GlobalTensor<half> w;
  w.SetGlobalBuffer((__gm__ half*)w_handle);
  AscendC::GlobalTensor<float> g;
  g.SetGlobalBuffer((__gm__ float*)g_handle);
  AscendC::GlobalTensor<half> v_new;
  v_new.SetGlobalBuffer((__gm__ half*)v_new_handle);
  AscendC::GlobalTensor<half> h0;
  h0.SetGlobalBuffer((__gm__ half*)h0_handle);
  AscendC::GlobalTensor<half> ht;
  ht.SetGlobalBuffer((__gm__ half*)ht_handle);
  AscendC::GlobalTensor<int> cu_seqlens;
  cu_seqlens.SetGlobalBuffer((__gm__ int*)cu_seqlens_handle);
  AscendC::GlobalTensor<float> ws_wh;
  ws_wh.SetGlobalBuffer((__gm__ float*)ws_wh_handle);
  AscendC::GlobalTensor<half> ws_vnew;
  ws_vnew.SetGlobalBuffer((__gm__ half*)ws_vnew_handle);
  AscendC::GlobalTensor<half> ws_hupd;
  ws_hupd.SetGlobalBuffer((__gm__ half*)ws_hupd_handle);
  AscendC::GlobalTensor<half> ws_h;
  ws_h.SetGlobalBuffer((__gm__ half*)ws_h_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto h_state_l1 = ascend_l1.GetWithOffset<half>(16384, 0);
  auto w_chunk_l1 = ascend_l1.GetWithOffset<half>(8192, 32768);
  auto wh_frag = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto v_new_l1 = ascend_l1.GetWithOffset<half>(8192, 49152);
  auto k_chunk_l1 = ascend_l1.GetWithOffset<half>(8192, 65536);
  auto hupd_frag = ascend_l0c.GetWithOffset<float>(16384, 32768);
  auto h_state_ub = ascend_ub.GetWithOffset<half>(8192, 0);
  auto wh_ub_float = ascend_ub.GetWithOffset<float>(4096, 16384);
  auto v_chunk_ub = ascend_ub.GetWithOffset<half>(4096, 32768);
  auto v_chunk_ub_float = ascend_ub.GetWithOffset<float>(4096, 40960);
  auto v_new_ub_float = ascend_ub.GetWithOffset<float>(4096, 57344);
  auto g_chunk_ub_all = ascend_ub.GetWithOffset<float>(64, 73728);
  auto g_chunk_ub = ascend_ub.GetWithOffset<float>(32, 73984);
  auto g_last_scalar = ascend_ub.GetWithOffset<float>(1, 74112);
  auto g_exp_ub = ascend_ub.GetWithOffset<float>(32, 74144);
  auto g_exp_ub_pad = ascend_ub.GetWithOffset<float>(64, 74272);
  auto g_mask_ub_pad = ascend_ub.GetWithOffset<uint8_t>(8, 74528);
  auto g_exp_ub_broc = ascend_ub.GetWithOffset<float>(4096, 74560);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(4096, 90944);
  auto h_state_ub_float = ascend_ub.GetWithOffset<float>(8192, 95040);
  auto v_new_ub = ascend_ub.GetWithOffset<half>(4096, 127808);
  auto hupd_ub = ascend_ub.GetWithOffset<half>(8192, 136000);
  auto hupd_ub_float = ascend_ub.GetWithOffset<float>(8192, 152384);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    AscendC::PipeBarrier<PIPE_ALL>();
    int32_t bos = cu_seqlens.GetValue(0);
    AscendC::PipeBarrier<PIPE_ALL>();
    int32_t eos = cu_seqlens.GetValue(1);
    for (int32_t i = 0; i < 32; ++i) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (i < (((eos + 63) - bos) / 64)) {
        tl::ascend::copy_gm_to_l1<half, 128, 128>(h_state_l1[0], ws_h[(cid * 16384)], 128, 128, 128);
        tl::ascend::copy_gm_to_l1<half, 64, 128>(w_chunk_l1[0], w[(((i * 65536) + (bos * 1024)) + (cid * 128))], 1024, ((-2048 <= ((0 - bos) - (i * 64))) ? 64 : ((-2112 < ((0 - bos) - (i * 64))) ? ((2112 - bos) - (i * 64)) : 0)), 128);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
        tl::ascend::gemm_v0<half, float, 64, 128, 128, false, false>(w_chunk_l1[0], h_state_l1[0], wh_frag[0], ascend_l0a, ascend_l0b, (bool)1);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 0>(ws_wh[(cid * 8192)], wh_frag[0], 128, 64, 128);
        tl::ascend::copy_gm_to_l1<half, 64, 128>(v_new_l1[0], ws_vnew[(cid * 8192)], 128, 64, 128);
        tl::ascend::copy_gm_to_l1<half, 64, 128>(k_chunk_l1[0], k[(((i * 32768) + (bos * 512)) + ((cid / 2) * 128))], 512, ((-2048 <= ((0 - bos) - (i * 64))) ? 64 : ((-2112 < ((0 - bos) - (i * 64))) ? ((2112 - bos) - (i * 64)) : 0)), 128);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
        tl::ascend::gemm_v0<half, float, 128, 128, 64, true, false>(k_chunk_l1[0], v_new_l1[0], hupd_frag[0], ascend_l0a, ascend_l0b, (bool)1);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
        tl::ascend::copy_l0c_to_gm<float, half, layout::RowMajor, 128, 128, 0>(ws_hupd[(cid * 16384)], hupd_frag[0], 128, 128, 128);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
  if ASCEND_IS_AIV {
    AscendC::PipeBarrier<PIPE_ALL>();
    int32_t bos_1 = cu_seqlens.GetValue(0);
    AscendC::PipeBarrier<PIPE_ALL>();
    int32_t eos_1 = cu_seqlens.GetValue(1);
    tl::ascend::copy_gm_to_ub<half, 128, 64>(h_state_ub[0], h0[((cid * 16384) + (vid * 8192))], 128, 64, 128, half(0.000000e+00f));
    for (int32_t i_1 = 0; i_1 < 32; ++i_1) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (i_1 < (((eos_1 + 63) - bos_1) / 64)) {
        tl::ascend::copy_ub_to_gm<half, 128, 64>(ws_h[((cid * 16384) + (vid * 8192))], h_state_ub[0], 131072, 1, 128);
        tl::ascend::copy_gm_to_ub<float, 128, 32>(wh_ub_float[0], ws_wh[((cid * 8192) + (vid * 4096))], 128, 32, 128, 0.000000e+00f);
        tl::ascend::copy_gm_to_ub<half, 128, 32>(v_chunk_ub[0], v[((((i_1 * 65536) + (vid * 32768)) + (bos_1 * 1024)) + (cid * 128))], 1024, ((-2080 <= (((0 - bos_1) - (vid * 32)) - (i_1 * 64))) ? 32 : ((-2112 < (((0 - bos_1) - (vid * 32)) - (i_1 * 64))) ? (((2112 - bos_1) - (vid * 32)) - (i_1 * 64)) : 0)), 128, half(0.000000e+00f));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        tl::ascend::copy_ub_to_ub<float, half, 4096>(v_chunk_ub_float[0], v_chunk_ub[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(v_new_ub_float[0], v_chunk_ub_float[0], wh_ub_float[0], 4096);
        tl::ascend::copy_gm_to_ub<float, 64>(g_chunk_ub_all[0], g[(((i_1 * 512) + (bos_1 * 8)) + cid)], 8, ((-2048 <= ((0 - bos_1) - (i_1 * 64))) ? 64 : ((-2112 < ((0 - bos_1) - (i_1 * 64))) ? ((2112 - bos_1) - (i_1 * 64)) : 0)), 1, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
        tl::ascend::copy_ub_to_ub<float, float, 32>(g_chunk_ub[0], g_chunk_ub_all[(vid * 32)]);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (((i_1 * 64) + 64) <= (eos_1 - bos_1)) {
          g_last_scalar.SetValue(0, g_chunk_ub_all.GetValue(63));
        } else {
          g_last_scalar.SetValue(0, g_chunk_ub_all.GetValue((((((int64_t)eos_1) - ((int64_t)bos_1)) - (((int64_t)i_1) * (int64_t)64)) - (int64_t)1)));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::Fill<float>(g_exp_ub[0], g_last_scalar.GetValue(0), 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(g_exp_ub[0], g_exp_ub[0], g_chunk_ub[0], 32);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, float, 32>(g_exp_ub_pad[0], g_exp_ub[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(g_mask_ub_pad[0], g_exp_ub_pad[0], 0.000000e+00f, AscendC::CMPMODE::LE, 64);
        AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(g_exp_ub_pad[0], g_mask_ub_pad[0], g_exp_ub_pad[0], static_cast<float>(-CUDART_INF_F), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, float, 32>(g_exp_ub[0], g_exp_ub_pad[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(g_exp_ub[0], g_exp_ub[0], 32);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::Broadcast<float, 2, 1, false>(g_exp_ub_broc[0],g_exp_ub[0],tmp_ub,(uint32_t[]){32, 128}, (uint32_t[]){32, 1});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(v_new_ub_float[0], v_new_ub_float[0], g_exp_ub_broc[0], 4096);
        AscendC::Exp(g_last_scalar[0], g_last_scalar[0], 1);
        tl::ascend::copy_ub_to_ub<float, half, 8192>(h_state_ub_float[0], h_state_ub[0]);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::PipeBarrier<PIPE_ALL>();
        auto g_last_scalar_scalar = g_last_scalar.GetValue(0);
        AscendC::Muls(h_state_ub_float[0], h_state_ub_float[0], g_last_scalar_scalar, 8192);
        }
        tl::ascend::copy_ub_to_ub<half, float, 4096>(v_new_ub[0], v_new_ub_float[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        tl::ascend::copy_ub_to_gm<half, 128, 32>(v_new[((((i_1 * 65536) + (vid * 32768)) + (bos_1 * 1024)) + (cid * 128))], v_new_ub[0], 1024, ((-2080 <= (((0 - bos_1) - (vid * 32)) - (i_1 * 64))) ? 32 : ((-2112 < (((0 - bos_1) - (vid * 32)) - (i_1 * 64))) ? (((2112 - bos_1) - (vid * 32)) - (i_1 * 64)) : 0)), 128);
        tl::ascend::copy_ub_to_gm<half, 128, 32>(ws_vnew[((cid * 8192) + (vid * 4096))], v_new_ub[0], 65536, 1, 128);
        tl::ascend::copy_gm_to_ub<half, 128, 64>(hupd_ub[0], ws_hupd[((cid * 16384) + (vid * 8192))], 128, 64, 128, half(0.000000e+00f));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        tl::ascend::copy_ub_to_ub<float, half, 8192>(hupd_ub_float[0], hupd_ub[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(h_state_ub_float[0], h_state_ub_float[0], hupd_ub_float[0], 8192);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<half, float, 8192>(h_state_ub[0], h_state_ub_float[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
        tl::ascend::copy_ub_to_gm<half, 128, 64>(h[(((i_1 * 131072) + (cid * 16384)) + (vid * 8192))], h_state_ub[0], 128, 64, 128);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    tl::ascend::copy_ub_to_gm<half, 128, 64>(ht[((cid * 16384) + (vid * 8192))], h_state_ub[0], 128, 64, 128);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* h_handle, uint8_t* k_handle, uint8_t* v_handle, uint8_t* w_handle, uint8_t* g_handle, uint8_t* v_new_handle, uint8_t* h0_handle, uint8_t* ht_handle, uint8_t* cu_seqlens_handle, uint8_t* ws_wh_handle, uint8_t* ws_vnew_handle, uint8_t* ws_hupd_handle, uint8_t* ws_h_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<8, nullptr, stream>>>(h_handle, k_handle, v_handle, w_handle, g_handle, v_new_handle, h0_handle, ht_handle, cu_seqlens_handle, ws_wh_handle, ws_vnew_handle, ws_hupd_handle, ws_h_handle, fftsAddr);
}
