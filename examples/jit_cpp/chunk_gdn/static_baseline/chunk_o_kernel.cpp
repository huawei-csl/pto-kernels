#include "common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *Q_handle, __gm__ half *K_handle, __gm__ half *V_handle, __gm__ half *S_handle, __gm__ float *G_handle, __gm__ float *Msk_handle, __gm__ half *workspace_1_handle, __gm__ half *workspace_2_handle, __gm__ half *workspace_3_handle, __gm__ half *O_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> q_l1;
  TASSIGN(q_l1, 0);
  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> k_l1;
  TASSIGN(k_l1, 32768);
  TileAcc<float, 128, 128, 128, 128> qk_l0;
  TASSIGN(qk_l0, 0);
  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> s_l1;
  TASSIGN(s_l1, 65536);
  TileAcc<float, 128, 128, 128, 128> qs_l0;
  TASSIGN(qs_l0, 65536);
  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> qk_l1;
  TASSIGN(qk_l1, 98304);
  chunk_gdn_pto::TileMatL1<half, 128, 128, 128, 128> v_l1;
  TASSIGN(v_l1, 131072);
  TileAcc<float, 128, 128, 128, 128> qkv_l0;
  TASSIGN(qkv_l0, 0);
  chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> g_ub;
  TASSIGN(g_ub, 0);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> msk_ub;
  TASSIGN(msk_ub, 512);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> qk_ub;
  TASSIGN(qk_ub, 33280);
  chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> g_v_ub;
  TASSIGN(g_v_ub, 66048);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> coeff_ub;
  TASSIGN(coeff_ub, 66304);
  chunk_gdn_pto::TileUbDataND<half, 64, 128, 64, 128> qk_ub_half;
  TASSIGN(qk_ub_half, 99072);
  chunk_gdn_pto::TileUbDataND<half, 64, 128, 64, 128> qs_ub_half;
  TASSIGN(qs_ub_half, 115456);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> qs_ub;
  TASSIGN(qs_ub, 131840);
  chunk_gdn_pto::TileUbDataND<half, 64, 128, 64, 128> o_ub_half;
  TASSIGN(o_ub_half, 164608);
  chunk_gdn_pto::TileUbDataND<float, 64, 128, 64, 128> o_ub;
  TASSIGN(o_ub, 512);
  auto vid = get_subblockid();
#if defined(__DAV_C220_CUBE__)
    chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(Q_handle + (cid * 16384), 0, 0, 128, 128);
    chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(K_handle + (cid * 16384), 32768, 0, 128, 128);
    chunk_gdn_pto::gemm_v0<half, float, 128, 128, 128, 128, 128, 128, 128, false, true>(q_l1, k_l1, qk_l0, (bool)1);
    chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(Q_handle + (cid * 16384), 0, 0, 128, 128);
    chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 33554432, 2097152, 16384, 128, 1, 128, 128>(S_handle + (cid * 16384), 65536, 0, 128, 128);
    chunk_gdn_pto::gemm_v0<half, float, 128, 128, 128, 128, 128, 128, 128, false, false>(q_l1, s_l1, qs_l0, (bool)1);
    chunk_gdn_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 1, 67108864, 16384, 128, 1, 128, 128>(workspace_1_handle + (cid * 16384), 0, 0, 128, 128);
    chunk_gdn_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 1, 67108864, 16384, 128, 1, 128, 128>(workspace_2_handle + (cid * 16384), 65536, 0, 128, 128);
    chunk_gdn_pto::set_cross_flag<PIPE_FIX>(0, 2);
    chunk_gdn_pto::wait_cross_flag(1);
    chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 1, 67108864, 16384, 128, 1, 128, 128>(workspace_3_handle + (cid * 16384), 98304, 0, 128, 128);
    chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(V_handle + (cid * 16384), 131072, 0, 128, 128);
    chunk_gdn_pto::gemm_v0<half, float, 128, 128, 128, 128, 128, 128, 128, false, false>(qk_l1, v_l1, qkv_l0, (bool)1);
    chunk_gdn_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 1, 67108864, 16384, 128, 1, 128, 128>(workspace_2_handle + (cid * 16384), 0, 0, 128, 128);
    chunk_gdn_pto::set_cross_flag<PIPE_FIX>(2, 2);
#endif
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    chunk_gdn_pto::copy_gm_to_ub<float, float, 1, 1, 1, 1, 128, 1, 1, 1, 524288, 1, 1, 128, pto::PadValue::Zero>(G_handle + (cid * 128), 0, 0, 1, 128);
    chunk_gdn_pto::copy_gm_to_ub<float, float, 1, 1, 1, 64, 128, 1, 1, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(Msk_handle + (vid * 8192), 512, 0, 64, 128);
    chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(qk_ub, 0.000000e+00f);
    chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> g_ub_temp_0;
    TASSIGN(g_ub_temp_0, 0 + (vid * 64) * 4);
    TMOV(g_v_ub, g_ub_temp_0);

  for (int32_t i = 0; i < 16; ++i) {
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      auto g_v_ub_scalar_temp_0 = g_v_ub.GetValue((i * 4));
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> g_ub_temp_1;
      TASSIGN(g_ub_temp_1, 0 + 0 * 4);
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> coeff_ub_temp_0;
      TASSIGN(coeff_ub_temp_0, 66304 + (i * 512) * 4);
      TADDS(coeff_ub_temp_0, g_ub_temp_1, -g_v_ub_scalar_temp_0);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      auto g_v_ub_scalar_temp_1 = g_v_ub.GetValue(((i * 4) + 1));
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> g_ub_temp_2;
      TASSIGN(g_ub_temp_2, 0 + 0 * 4);
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> coeff_ub_temp_1;
      TASSIGN(coeff_ub_temp_1, 66304 + ((i * 512) + 128) * 4);
      TADDS(coeff_ub_temp_1, g_ub_temp_2, -g_v_ub_scalar_temp_1);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      auto g_v_ub_scalar_temp_2 = g_v_ub.GetValue(((i * 4) + 2));
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> g_ub_temp_3;
      TASSIGN(g_ub_temp_3, 0 + 0 * 4);
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> coeff_ub_temp_2;
      TASSIGN(coeff_ub_temp_2, 66304 + ((i * 512) + 256) * 4);
      TADDS(coeff_ub_temp_2, g_ub_temp_3, -g_v_ub_scalar_temp_2);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      auto g_v_ub_scalar_temp_3 = g_v_ub.GetValue(((i * 4) + 3));
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> g_ub_temp_4;
      TASSIGN(g_ub_temp_4, 0 + 0 * 4);
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> coeff_ub_temp_3;
      TASSIGN(coeff_ub_temp_3, 66304 + ((i * 512) + 384) * 4);
      TADDS(coeff_ub_temp_3, g_ub_temp_4, -g_v_ub_scalar_temp_3);
    }
    TSUB(coeff_ub, qk_ub, coeff_ub);
    TMUL(coeff_ub, coeff_ub, msk_ub);
    TEXP(coeff_ub, coeff_ub);
    TEXP(g_v_ub, g_v_ub);
    chunk_gdn_pto::wait_cross_flag(0);
    chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 1, 67108864, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(workspace_1_handle + ((cid * 16384) + (vid * 8192)), 99072, 0, 64, 128);
    chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);
    chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE2> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE2> (0);
    chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 1, 67108864, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(workspace_2_handle + ((cid * 16384) + (vid * 8192)), 115456, 0, 64, 128);
    TMUL(qk_ub, qk_ub, coeff_ub);
    TMUL(qk_ub, qk_ub, msk_ub);
    TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);
    chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 1, 67108864, 16384, 128, 1, 64, 128>(workspace_3_handle + ((cid * 16384) + (vid * 8192)), 99072, 0, 64, 128);
    chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);
    chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);

  for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      auto g_v_ub_scalar_temp_4 = g_v_ub.GetValue((i_1 * 4));
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> qs_ub_temp_0;
      TASSIGN(qs_ub_temp_0, 131840 + (i_1 * 512) * 4);
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> qs_ub_temp_1;
      TASSIGN(qs_ub_temp_1, 131840 + (i_1 * 512) * 4);
      TMULS(qs_ub_temp_1, qs_ub_temp_0, g_v_ub_scalar_temp_4);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      auto g_v_ub_scalar_temp_5 = g_v_ub.GetValue(((i_1 * 4) + 1));
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> qs_ub_temp_2;
      TASSIGN(qs_ub_temp_2, 131840 + ((i_1 * 512) + 128) * 4);
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> qs_ub_temp_3;
      TASSIGN(qs_ub_temp_3, 131840 + ((i_1 * 512) + 128) * 4);
      TMULS(qs_ub_temp_3, qs_ub_temp_2, g_v_ub_scalar_temp_5);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      auto g_v_ub_scalar_temp_6 = g_v_ub.GetValue(((i_1 * 4) + 2));
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> qs_ub_temp_4;
      TASSIGN(qs_ub_temp_4, 131840 + ((i_1 * 512) + 256) * 4);
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> qs_ub_temp_5;
      TASSIGN(qs_ub_temp_5, 131840 + ((i_1 * 512) + 256) * 4);
      TMULS(qs_ub_temp_5, qs_ub_temp_4, g_v_ub_scalar_temp_6);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      auto g_v_ub_scalar_temp_7 = g_v_ub.GetValue(((i_1 * 4) + 3));
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> qs_ub_temp_6;
      TASSIGN(qs_ub_temp_6, 131840 + ((i_1 * 512) + 384) * 4);
      chunk_gdn_pto::TileUbDataND<float, 1, 128, 1, 128> qs_ub_temp_7;
      TASSIGN(qs_ub_temp_7, 131840 + ((i_1 * 512) + 384) * 4);
      TMULS(qs_ub_temp_7, qs_ub_temp_6, g_v_ub_scalar_temp_7);
    }
    chunk_gdn_pto::wait_cross_flag(2);
    chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 1, 67108864, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(workspace_2_handle + ((cid * 16384) + (vid * 8192)), 164608, 0, 64, 128);
    chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
    TADD(o_ub, qs_ub, o_ub);
    TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);
    chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128>(O_handle + ((cid * 16384) + (vid * 8192)), 164608, 0, 64, 128);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *Q_handle, __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle, __gm__ uint8_t *S_handle, __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle, __gm__ uint8_t *workspace_1_handle, __gm__ uint8_t *workspace_2_handle, __gm__ uint8_t *workspace_3_handle, __gm__ uint8_t *O_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(Q_handle),
     reinterpret_cast<__gm__ half *>(K_handle),
     reinterpret_cast<__gm__ half *>(V_handle),
     reinterpret_cast<__gm__ half *>(S_handle),
     reinterpret_cast<__gm__ float *>(G_handle),
     reinterpret_cast<__gm__ float *>(Msk_handle),
     reinterpret_cast<__gm__ half *>(workspace_1_handle),
     reinterpret_cast<__gm__ half *>(workspace_2_handle),
     reinterpret_cast<__gm__ half *>(workspace_3_handle),
     reinterpret_cast<__gm__ half *>(O_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *Q_handle, uint8_t *K_handle, uint8_t *V_handle, uint8_t *S_handle, uint8_t *G_handle, uint8_t *Msk_handle, uint8_t *workspace_1_handle, uint8_t *workspace_2_handle, uint8_t *workspace_3_handle, uint8_t *O_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<4096, nullptr, stream>>>(Q_handle, K_handle, V_handle, S_handle, G_handle, Msk_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, O_handle, fftsAddr);
}
