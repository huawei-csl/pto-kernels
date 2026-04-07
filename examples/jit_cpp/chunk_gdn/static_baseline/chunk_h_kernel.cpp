#include "common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *K_handle, __gm__ half *W_handle, __gm__ half *U_handle, __gm__ float *G_handle, __gm__ half *workspace_1_handle, __gm__ half *workspace_2_handle, __gm__ half *workspace_3_handle, __gm__ half *workspace_4_handle, __gm__ half *S_handle, __gm__ half *V_handle, __gm__ half *FS_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> s_l1;
  TASSIGN(s_l1, 0);
  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> w_l1;
  TASSIGN(w_l1, 32768);
  TileAcc<float, 128, 128, 128, 128> ws_l0;
  TASSIGN(ws_l0, 0);
  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> k_l1;
  TASSIGN(k_l1, 65536);
  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> v_l1;
  TASSIGN(v_l1, 98304);
  TileAcc<float, 128, 128, 128, 128> kv_l0;
  TASSIGN(kv_l0, 65536);
  chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> zero_ub;
  TASSIGN(zero_ub, 0);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> s_ub;
  TASSIGN(s_ub, 256);
  chunk_gdn_pto::TileUbDataND<half, 64, 128, 64, 128> k_ub_half;
  TASSIGN(k_ub_half, 33024);
  chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> g_ub;
  TASSIGN(g_ub, 49408);
  chunk_gdn_pto::TileUbDataND<half, 64, 128, 64, 128> s_ub_half;
  TASSIGN(s_ub_half, 165120);
  chunk_gdn_pto::TileUbDataND<half, 64, 128, 64, 128> u_ub_half;
  TASSIGN(u_ub_half, 49920);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> k_ub;
  TASSIGN(k_ub, 66304);
  chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> g_v_ub;
  TASSIGN(g_v_ub, 99072);
  chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> coeff_ub;
  TASSIGN(coeff_ub, 99328);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> u_ub;
  TASSIGN(u_ub, 99584);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> ws_ub;
  TASSIGN(ws_ub, 132352);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> kv_ub;
  TASSIGN(kv_ub, 49920);
  auto vid = get_subblockid();
#if defined(__DAV_C220_CUBE__)

  for (int32_t i = 0; i < 128; ++i) {
      chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 1, 524288, 16384, 128, 1, 128, 128>(workspace_3_handle + (cid * 16384), 0, 0, 128, 128);
      chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(W_handle + ((cid * 2097152) + (i * 16384)), 32768, 0, 128, 128);
      chunk_gdn_pto::gemm_v0<half, float, 128, 128, 128, 128, 128, 128, 128, false, false>(w_l1, s_l1, ws_l0, (bool)1);
      chunk_gdn_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 1, 524288, 16384, 128, 1, 128, 128>(workspace_1_handle + (cid * 16384), 0, 0, 128, 128);
      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(0, 2);
      chunk_gdn_pto::wait_cross_flag(1);
      chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 1, 524288, 16384, 128, 1, 128, 128>(workspace_2_handle + (cid * 16384), 65536, 0, 128, 128);
      chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(V_handle + ((cid * 2097152) + (i * 16384)), 98304, 0, 128, 128);
      chunk_gdn_pto::gemm_v0<half, float, 128, 128, 128, 128, 128, 128, 128, true, false>(k_l1, v_l1, kv_l0, (bool)1);
      chunk_gdn_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 1, 524288, 16384, 128, 1, 128, 128>(workspace_4_handle + (cid * 16384), 65536, 0, 128, 128);
      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(2, 2);
      chunk_gdn_pto::wait_cross_flag(3);
    }
#endif
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(zero_ub, 0.000000e+00f);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(s_ub, 0.000000e+00f);
    chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128, pto::PadValue::Zero>(K_handle + ((cid * 2097152) + (vid * 8192)), 33024, 0, 64, 128);
    chunk_gdn_pto::copy_gm_to_ub<float, float, 1, 1, 1, 1, 128, 1, 1, 1, 524288, 1, 1, 128, pto::PadValue::Zero>(G_handle + (cid * 16384), 49408, 0, 1, 128);
    chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_S> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_S> (0);

  for (int32_t i_1 = 0; i_1 < 128; ++i_1) {
      chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128, pto::PadValue::Zero>(U_handle + (((cid * 2097152) + (i_1 * 16384)) + (vid * 8192)), 49920, 0, 64, 128);
      TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);
      chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> g_ub_temp_0;
      TASSIGN(g_ub_temp_0, 49408 + (vid * 64) * 4);
      TMOV(g_v_ub, g_ub_temp_0);
      float tmp = g_ub.GetValue(127);
      TADDS(coeff_ub, g_v_ub, -tmp);
      pipe_barrier(PIPE_V);
      TSUB(coeff_ub, zero_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);
      TEXP(g_ub, g_ub);
      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
      TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);

  for (int32_t i_2 = 0; i_2 < 16; ++i_2) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto coeff_ub_scalar_temp_0 = coeff_ub.GetValue((i_2 * 4));
        chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> k_ub_temp_0;
        TASSIGN(k_ub_temp_0, 66304 + (i_2 * 512) * 4);
        chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> k_ub_temp_1;
        TASSIGN(k_ub_temp_1, 66304 + (i_2 * 512) * 4);
        TMULS(k_ub_temp_1, k_ub_temp_0, coeff_ub_scalar_temp_0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto coeff_ub_scalar_temp_1 = coeff_ub.GetValue(((i_2 * 4) + 1));
        chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> k_ub_temp_2;
        TASSIGN(k_ub_temp_2, 66304 + ((i_2 * 512) + 128) * 4);
        chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> k_ub_temp_3;
        TASSIGN(k_ub_temp_3, 66304 + ((i_2 * 512) + 128) * 4);
        TMULS(k_ub_temp_3, k_ub_temp_2, coeff_ub_scalar_temp_1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto coeff_ub_scalar_temp_2 = coeff_ub.GetValue(((i_2 * 4) + 2));
        chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> k_ub_temp_4;
        TASSIGN(k_ub_temp_4, 66304 + ((i_2 * 512) + 256) * 4);
        chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> k_ub_temp_5;
        TASSIGN(k_ub_temp_5, 66304 + ((i_2 * 512) + 256) * 4);
        TMULS(k_ub_temp_5, k_ub_temp_4, coeff_ub_scalar_temp_2);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto coeff_ub_scalar_temp_3 = coeff_ub.GetValue(((i_2 * 4) + 3));
        chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> k_ub_temp_6;
        TASSIGN(k_ub_temp_6, 66304 + ((i_2 * 512) + 384) * 4);
        chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> k_ub_temp_7;
        TASSIGN(k_ub_temp_7, 66304 + ((i_2 * 512) + 384) * 4);
        TMULS(k_ub_temp_7, k_ub_temp_6, coeff_ub_scalar_temp_3);
      }
      chunk_gdn_pto::wait_cross_flag(0);
      chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 1, 524288, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(workspace_1_handle + ((cid * 16384) + (vid * 8192)), 49920, 0, 64, 128);
      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
      TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
      TSUB(u_ub, u_ub, ws_ub);
      TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
      chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128>(V_handle + (((cid * 2097152) + (i_1 * 16384)) + (vid * 8192)), 49920, 0, 64, 128);
      chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 1, 524288, 16384, 128, 1, 64, 128>(workspace_2_handle + ((cid * 16384) + (vid * 8192)), 33024, 0, 64, 128);
      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);
      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE3, PIPE_S> (0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE3, PIPE_S> (0);
      float tmp_1 = g_ub.GetValue(127);
      TMULS(s_ub, s_ub, tmp_1);
      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE2> (0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE2> (0);
      if (i_1 < 127) {
        chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128, pto::PadValue::Zero>(K_handle + ((((cid * 2097152) + (i_1 * 16384)) + (vid * 8192)) + 16384), 33024, 0, 64, 128);
        chunk_gdn_pto::copy_gm_to_ub<float, float, 1, 1, 1, 1, 128, 1, 1, 1, 524288, 1, 1, 128, pto::PadValue::Zero>(G_handle + (((cid * 16384) + (i_1 * 128)) + 128), 49408, 0, 1, 128);
      }
      chunk_gdn_pto::wait_cross_flag(2);
      chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 1, 524288, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(workspace_4_handle + ((cid * 16384) + (vid * 8192)), 165120, 0, 64, 128);
      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
      TCVT(kv_ub, s_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_ALL);
      TADD(s_ub, s_ub, kv_ub);
      TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
      if (i_1 < 127) {
        chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
        chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
        chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 1, 524288, 16384, 128, 1, 64, 128>(workspace_3_handle + ((cid * 16384) + (vid * 8192)), 165120, 0, 64, 128);
        chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 33554432, 2097152, 16384, 128, 1, 64, 128>(S_handle + ((((cid * 2097152) + (i_1 * 16384)) + (vid * 8192)) + 16384), 165120, 0, 64, 128);
      }
      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(3, 2);
    }
    chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 524288, 262144, 16384, 128, 1, 64, 128>(FS_handle + ((cid * 16384) + (vid * 8192)), 165120, 0, 64, 128);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *K_handle, __gm__ uint8_t *W_handle, __gm__ uint8_t *U_handle, __gm__ uint8_t *G_handle, __gm__ uint8_t *workspace_1_handle, __gm__ uint8_t *workspace_2_handle, __gm__ uint8_t *workspace_3_handle, __gm__ uint8_t *workspace_4_handle, __gm__ uint8_t *S_handle, __gm__ uint8_t *V_handle, __gm__ uint8_t *FS_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(K_handle),
     reinterpret_cast<__gm__ half *>(W_handle),
     reinterpret_cast<__gm__ half *>(U_handle),
     reinterpret_cast<__gm__ float *>(G_handle),
     reinterpret_cast<__gm__ half *>(workspace_1_handle),
     reinterpret_cast<__gm__ half *>(workspace_2_handle),
     reinterpret_cast<__gm__ half *>(workspace_3_handle),
     reinterpret_cast<__gm__ half *>(workspace_4_handle),
     reinterpret_cast<__gm__ half *>(S_handle),
     reinterpret_cast<__gm__ half *>(V_handle),
     reinterpret_cast<__gm__ half *>(FS_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *K_handle, uint8_t *W_handle, uint8_t *U_handle, uint8_t *G_handle, uint8_t *workspace_1_handle, uint8_t *workspace_2_handle, uint8_t *workspace_3_handle, uint8_t *workspace_4_handle, uint8_t *S_handle, uint8_t *V_handle, uint8_t *FS_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<32, nullptr, stream>>>(K_handle, W_handle, U_handle, G_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_4_handle, S_handle, V_handle, FS_handle, fftsAddr);
}
