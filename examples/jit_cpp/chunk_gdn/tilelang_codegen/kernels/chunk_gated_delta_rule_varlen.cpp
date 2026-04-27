#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *h_handle, __gm__ half *k_handle, __gm__ half *v_handle, __gm__ half *w_handle, __gm__ float *g_handle, __gm__ half *v_new_handle, __gm__ half *h0_handle, __gm__ half *ht_handle, __gm__ int *cu_seqlens_handle, __gm__ float *ws_wh_handle, __gm__ half *ws_vnew_handle, __gm__ half *ws_hupd_handle, __gm__ half *ws_h_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileMatL1<half, 128, 128, 128, 128> h_state_l1;
  TASSIGN(h_state_l1, 0);
  tl::ascend_pto::TileMatL1<half, 64, 128, 64, 128> w_chunk_l1;
  TASSIGN(w_chunk_l1, 32768);
  TileAcc<float, 64, 128, 64, 128> wh_frag;
  TASSIGN(wh_frag, 0);
  tl::ascend_pto::TileMatL1<half, 64, 128, 64, 128> v_new_l1;
  TASSIGN(v_new_l1, 49152);
  tl::ascend_pto::TileMatL1<half, 64, 128, 64, 128> k_chunk_l1;
  TASSIGN(k_chunk_l1, 65536);
  TileAcc<float, 128, 128, 128, 128> hupd_frag;
  TASSIGN(hupd_frag, 32768);
  tl::ascend_pto::TileUbDataND<half, 64, 128, 64, 128> h_state_ub;
  TASSIGN(h_state_ub, 0);
  tl::ascend_pto::TileUbDataND<float, 32, 128, 32, 128> wh_ub_float;
  TASSIGN(wh_ub_float, 16384);
  tl::ascend_pto::TileUbDataND<half, 32, 128, 32, 128> v_chunk_ub;
  TASSIGN(v_chunk_ub, 32768);
  tl::ascend_pto::TileUbDataND<float, 32, 128, 32, 128> v_chunk_ub_float;
  TASSIGN(v_chunk_ub_float, 40960);
  tl::ascend_pto::TileUbDataND<float, 32, 128, 32, 128> v_new_ub_float;
  TASSIGN(v_new_ub_float, 57344);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> g_chunk_ub_all;
  TASSIGN(g_chunk_ub_all, 73728);
  tl::ascend_pto::TileUbDataND<float, 1, 32, 1, 32> g_chunk_ub;
  TASSIGN(g_chunk_ub, 73984);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> g_last_scalar;
  TASSIGN(g_last_scalar, 74112);
  tl::ascend_pto::TileUbDataND<float, 1, 32, 1, 32> g_exp_ub;
  TASSIGN(g_exp_ub, 74144);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> g_exp_ub_pad;
  TASSIGN(g_exp_ub_pad, 74272);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 32, 1, 8> g_mask_ub_pad;
  TASSIGN(g_mask_ub_pad, 74528);
  tl::ascend_pto::TileUbDataND<float, 32, 128, 32, 128> g_exp_ub_broc;
  TASSIGN(g_exp_ub_broc, 74560);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 8192, 1, 8192> tmp_ub;
  TASSIGN(tmp_ub, 90944);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> h_state_ub_float;
  TASSIGN(h_state_ub_float, 99136);
  tl::ascend_pto::TileUbDataND<half, 32, 128, 32, 128> v_new_ub;
  TASSIGN(v_new_ub, 131904);
  tl::ascend_pto::TileUbDataND<half, 64, 128, 64, 128> hupd_ub;
  TASSIGN(hupd_ub, 140096);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> hupd_ub_float;
  TASSIGN(hupd_ub_float, 156480);
  auto vid = get_subblockid();
#if defined(__DAV_C220_CUBE__)
    pipe_barrier(PIPE_ALL);
    int32_t bos = *(cu_seqlens_handle + 0);
    pipe_barrier(PIPE_ALL);
    int32_t eos = *(cu_seqlens_handle + 1);

  for (int32_t i = 0; i < 32; ++i) {
      pipe_barrier(PIPE_ALL);
      if (i < (((eos + 63) - bos) / 64)) {
        tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 131072, 131072, 16384, 128, 1, 128, 128>(ws_h_handle + (cid * 16384), 0, 0, 128, 128);
        tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 1, 1, 2162688, 1024, 1, 64, 128>(w_handle + (((i * 65536) + (bos * 1024)) + (cid * 128)), 32768, 0, ((-2048 <= ((0 - bos) - (i * 64))) ? 64 : ((-2112 < ((0 - bos) - (i * 64))) ? ((2112 - bos) - (i * 64)) : 0)), 128);
        set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
        tl::ascend_pto::gemm_v0<half, float, 64, 128, 128, 64, 128, 128, 128, false, false>(w_chunk_l1, h_state_l1, wh_frag, (bool)1);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        tl::ascend_pto::copy_l0c_to_gm<float, float, 1, 1, 1, 64, 128, 65536, 65536, 8192, 128, 1, 64, 128>(ws_wh_handle + (cid * 8192), 0, 0, 64, 128);
        tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 65536, 65536, 8192, 128, 1, 64, 128>(ws_vnew_handle + (cid * 8192), 49152, 0, 64, 128);
        tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 1, 1, 1081344, 512, 1, 64, 128>(k_handle + (((i * 32768) + (bos * 512)) + ((cid / 2) * 128)), 65536, 0, ((-2048 <= ((0 - bos) - (i * 64))) ? 64 : ((-2112 < ((0 - bos) - (i * 64))) ? ((2112 - bos) - (i * 64)) : 0)), 128);
        set_flag(PIPE_MTE2, PIPE_M, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID3);
        tl::ascend_pto::gemm_v0<half, float, 128, 128, 64, 128, 128, 64, 64, true, false>(k_chunk_l1, v_new_l1, hupd_frag, (bool)1);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
        tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 131072, 131072, 16384, 128, 1, 128, 128>(ws_hupd_handle + (cid * 16384), 32768, 0, 128, 128);
      }
      pipe_barrier(PIPE_ALL);
      pipe_barrier(PIPE_ALL);
    }
#endif
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    pipe_barrier(PIPE_ALL);
    int32_t bos_1 = *(cu_seqlens_handle + 0);
    pipe_barrier(PIPE_ALL);
    int32_t eos_1 = *(cu_seqlens_handle + 1);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 131072, 131072, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(h0_handle + ((cid * 16384) + (vid * 8192)), 0, 0, 64, 128);

  for (int32_t i_1 = 0; i_1 < 32; ++i_1) {
      pipe_barrier(PIPE_ALL);
      if (i_1 < (((eos_1 + 63) - bos_1) / 64)) {
        tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 1, 1, 131072, 131072, 1, 64, 128>(ws_h_handle + ((cid * 16384) + (vid * 8192)), 0, 0, 1, 128);
        tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 32, 128, 65536, 65536, 8192, 128, 1, 32, 128, pto::PadValue::Zero>(ws_wh_handle + ((cid * 8192) + (vid * 4096)), 16384, 0, 32, 128);
        tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 32, 128, 1, 1, 2162688, 1024, 1, 32, 128, pto::PadValue::Zero>(v_handle + ((((i_1 * 65536) + (vid * 32768)) + (bos_1 * 1024)) + (cid * 128)), 32768, 0, ((-2080 <= (((0 - bos_1) - (vid * 32)) - (i_1 * 64))) ? 32 : ((-2112 < (((0 - bos_1) - (vid * 32)) - (i_1 * 64))) ? (((2112 - bos_1) - (vid * 32)) - (i_1 * 64)) : 0)), 128);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TCVT(v_chunk_ub_float, v_chunk_ub, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TSUB(v_new_ub_float, v_chunk_ub_float, wh_ub_float);
        tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 1, 64, 1, 1, 16896, 8, 1, 1, 64, pto::PadValue::Zero>(g_handle + (((i_1 * 512) + (bos_1 * 8)) + cid), 73728, 0, ((-2048 <= ((0 - bos_1) - (i_1 * 64))) ? 64 : ((-2112 < ((0 - bos_1) - (i_1 * 64))) ? ((2112 - bos_1) - (i_1 * 64)) : 0)), 1);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        tl::ascend_pto::TileUbDataND<float, 1, 32, 1, 32> g_chunk_ub_all_temp_0;
        TASSIGN(g_chunk_ub_all_temp_0, 73728 + (vid * 32) * 4);
        TMOV(g_chunk_ub, g_chunk_ub_all_temp_0);
        pipe_barrier(PIPE_ALL);
        if (((i_1 * 64) + 64) <= (eos_1 - bos_1)) {
          g_last_scalar.SetValue(0, g_chunk_ub_all.GetValue(63));
        } else {
          g_last_scalar.SetValue(0, g_chunk_ub_all.GetValue((((((int64_t)eos_1) - ((int64_t)bos_1)) - (((int64_t)i_1) * (int64_t)64)) - (int64_t)1)));
        }
        pipe_barrier(PIPE_ALL);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        TEXPANDS(g_exp_ub, g_last_scalar.GetValue(0));
        pipe_barrier(PIPE_V);
        TSUB(g_exp_ub, g_exp_ub, g_chunk_ub);
        pipe_barrier(PIPE_V);
        tl::ascend_pto::TileUbDataND<float, 1, 32, 1, 32> g_exp_ub_pad_temp_0;
        TASSIGN(g_exp_ub_pad_temp_0, 74272 + 0 * 4);
        TMOV(g_exp_ub_pad_temp_0, g_exp_ub);
        pipe_barrier(PIPE_V);
        tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> g_exp_ub_pad_temp_1;
        TASSIGN(g_exp_ub_pad_temp_1, 74272 + 0 * 4);
        tl::ascend_pto::TileUbDataND<uint8_t, 1, 32, 1, 8> g_mask_ub_pad_temp_0;
        TASSIGN(g_mask_ub_pad_temp_0, 74528 + 0 * 1);
        tl::ascend_pto::compare_scalar(g_mask_ub_pad_temp_0, g_exp_ub_pad_temp_1, 0.000000e+00f, CmpMode::LE);
        pipe_barrier(PIPE_V);
        TSELS(g_exp_ub_pad, g_mask_ub_pad, g_exp_ub_pad, -CUDART_INF_F);
        pipe_barrier(PIPE_V);
        tl::ascend_pto::TileUbDataND<float, 1, 32, 1, 32> g_exp_ub_pad_temp_2;
        TASSIGN(g_exp_ub_pad_temp_2, 74272 + 0 * 4);
        TMOV(g_exp_ub, g_exp_ub_pad_temp_2);
        pipe_barrier(PIPE_V);
        TEXP(g_exp_ub, g_exp_ub);
        pipe_barrier(PIPE_V);
        tl::ascend_pto::TileUbDataDN<float, 32, 1, 32, 1> g_exp_ub_temp_0;
        TASSIGN(g_exp_ub_temp_0, 74144 + 0 * 4);
        TROWEXPAND(g_exp_ub_broc, g_exp_ub_temp_0);
        pipe_barrier(PIPE_V);
        TMUL(v_new_ub_float, v_new_ub_float, g_exp_ub_broc);
        tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> g_last_scalar_temp_0;
        TASSIGN(g_last_scalar_temp_0, 74112 + 0 * 4);
        tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> g_last_scalar_temp_1;
        TASSIGN(g_last_scalar_temp_1, 74112 + 0 * 4);
        TEXP(g_last_scalar_temp_1, g_last_scalar_temp_0);
        TCVT(h_state_ub_float, h_state_ub, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto g_last_scalar_scalar_temp_0 = g_last_scalar.GetValue(0);
        TMULS(h_state_ub_float, h_state_ub_float, g_last_scalar_scalar_temp_0);
        TCVT(v_new_ub, v_new_ub_float, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 32, 128, 1, 1, 2162688, 1024, 1, 32, 128>(v_new_handle + ((((i_1 * 65536) + (vid * 32768)) + (bos_1 * 1024)) + (cid * 128)), 131904, 0, ((-2080 <= (((0 - bos_1) - (vid * 32)) - (i_1 * 64))) ? 32 : ((-2112 < (((0 - bos_1) - (vid * 32)) - (i_1 * 64))) ? (((2112 - bos_1) - (vid * 32)) - (i_1 * 64)) : 0)), 128);
        tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 32, 128, 1, 1, 65536, 65536, 1, 32, 128>(ws_vnew_handle + ((cid * 8192) + (vid * 4096)), 131904, 0, 1, 128);
        tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 131072, 131072, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(ws_hupd_handle + ((cid * 16384) + (vid * 8192)), 140096, 0, 64, 128);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        TCVT(hupd_ub_float, hupd_ub, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TADD(h_state_ub_float, h_state_ub_float, hupd_ub_float);
        pipe_barrier(PIPE_V);
        TCVT(h_state_ub, h_state_ub_float, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 4194304, 131072, 16384, 128, 1, 64, 128>(h_handle + (((i_1 * 131072) + (cid * 16384)) + (vid * 8192)), 0, 0, 64, 128);
      }
      pipe_barrier(PIPE_ALL);
      pipe_barrier(PIPE_ALL);
    }
    tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 131072, 131072, 16384, 128, 1, 64, 128>(ht_handle + ((cid * 16384) + (vid * 8192)), 0, 0, 64, 128);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *h_handle, __gm__ uint8_t *k_handle, __gm__ uint8_t *v_handle, __gm__ uint8_t *w_handle, __gm__ uint8_t *g_handle, __gm__ uint8_t *v_new_handle, __gm__ uint8_t *h0_handle, __gm__ uint8_t *ht_handle, __gm__ uint8_t *cu_seqlens_handle, __gm__ uint8_t *ws_wh_handle, __gm__ uint8_t *ws_vnew_handle, __gm__ uint8_t *ws_hupd_handle, __gm__ uint8_t *ws_h_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(h_handle),
     reinterpret_cast<__gm__ half *>(k_handle),
     reinterpret_cast<__gm__ half *>(v_handle),
     reinterpret_cast<__gm__ half *>(w_handle),
     reinterpret_cast<__gm__ float *>(g_handle),
     reinterpret_cast<__gm__ half *>(v_new_handle),
     reinterpret_cast<__gm__ half *>(h0_handle),
     reinterpret_cast<__gm__ half *>(ht_handle),
     reinterpret_cast<__gm__ int *>(cu_seqlens_handle),
     reinterpret_cast<__gm__ float *>(ws_wh_handle),
     reinterpret_cast<__gm__ half *>(ws_vnew_handle),
     reinterpret_cast<__gm__ half *>(ws_hupd_handle),
     reinterpret_cast<__gm__ half *>(ws_h_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *h_handle, uint8_t *k_handle, uint8_t *v_handle, uint8_t *w_handle, uint8_t *g_handle, uint8_t *v_new_handle, uint8_t *h0_handle, uint8_t *ht_handle, uint8_t *cu_seqlens_handle, uint8_t *ws_wh_handle, uint8_t *ws_vnew_handle, uint8_t *ws_hupd_handle, uint8_t *ws_h_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<8, nullptr, stream>>>(h_handle, k_handle, v_handle, w_handle, g_handle, v_new_handle, h0_handle, ht_handle, cu_seqlens_handle, ws_wh_handle, ws_vnew_handle, ws_hupd_handle, ws_h_handle, fftsAddr);
}
