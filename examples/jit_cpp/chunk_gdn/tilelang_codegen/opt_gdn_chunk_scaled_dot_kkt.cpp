#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *K_handle, __gm__ half *Beta_handle, __gm__ float *G_handle, __gm__ float *Msk_handle, __gm__ half *workspace_handle, __gm__ half *A_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileMatL1<half, 128, 128, 128, 128> k_l1;
  TASSIGN(k_l1, 0);
  TileAcc<float, 128, 128, 128, 128> a_l0;
  TASSIGN(a_l0, 0);
  tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> g_ub;
  TASSIGN(g_ub, 0);
  tl::ascend_pto::TileUbDataND<half, 1, 64, 1, 64> beta_ub_half;
  TASSIGN(beta_ub_half, 512);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> beta_ub;
  TASSIGN(beta_ub, 640);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> g_v_ub;
  TASSIGN(g_v_ub, 896);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> a_ub;
  TASSIGN(a_ub, 1152);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> g_r_ub;
  TASSIGN(g_r_ub, 33920);
  tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> g_c_ub;
  TASSIGN(g_c_ub, 34176);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> msk_ub;
  TASSIGN(msk_ub, 34688);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> g_r_2d_ub;
  TASSIGN(g_r_2d_ub, 67456);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 24576, 1, 24576> tmp_ub;
  TASSIGN(tmp_ub, 100224);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> g_c_2d_ub;
  TASSIGN(g_c_2d_ub, 124800);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> coeff_ub;
  TASSIGN(coeff_ub, 157568);
  tl::ascend_pto::TileUbDataND<half, 64, 128, 64, 128> a_ub_half;
  TASSIGN(a_ub_half, 67456);
  auto vid = get_subblockid();
#if defined(__DAV_C220_CUBE__)
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(K_handle + (cid * 16384), 0, 0, 128, 128);
    tl::ascend_pto::gemm_v0<half, float, 128, 128, 128, 128, 128, 128, 128, false, true>(k_l1, k_l1, a_l0, (bool)1);
    tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(workspace_handle + (cid * 16384), 0, 0, 128, 128);
    tl::ascend_pto::set_cross_flag<PIPE_FIX>(0, 2);
#endif
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 1, 128, 1, 1, 1, 524288, 1, 1, 128, pto::PadValue::Zero>(G_handle + (cid * 128), 0, 0, 1, 128);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 1, 64, 1, 1, 1, 524288, 1, 1, 64, pto::PadValue::Zero>(Beta_handle + ((cid * 128) + (vid * 64)), 512, 0, 1, 64);
    tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
    tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> g_ub_temp_0;
    TASSIGN(g_ub_temp_0, 0 + (vid * 64) * 4);
    TMOV(g_v_ub, g_ub_temp_0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(a_ub, 0.000000e+00f);
    TLOG(beta_ub, beta_ub);
    pipe_barrier(PIPE_V);
    TADD(g_v_ub, g_v_ub, beta_ub);
    pipe_barrier(PIPE_V);
    TMOV(g_r_ub, g_v_ub);
    TMOV(g_c_ub, g_ub);
    tl::ascend_pto::set_flag_pipeline<PIPE_V, PIPE_MTE2> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE2> (0);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 64, 128, 1, 1, 16384, 128, 1, 64, 128, pto::PadValue::Zero>(Msk_handle + (vid * 8192), 34688, 0, 64, 128);
    tl::ascend_pto::TileUbDataDN<float, 64, 1, 64, 1> g_r_ub_temp_0;
    TASSIGN(g_r_ub_temp_0, 33920 + 0 * 4);
    TROWEXPAND(g_r_2d_ub, g_r_ub_temp_0);
    TCOLEXPAND(g_c_2d_ub, g_c_ub);
    TSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);
    TEXP(coeff_ub, coeff_ub);
    tl::ascend_pto::set_flag_pipeline<PIPE_V, PIPE_MTE2> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE2> (0);
    tl::ascend_pto::wait_cross_flag(0);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128, pto::PadValue::Zero>(workspace_handle + ((cid * 16384) + (vid * 8192)), 67456, 0, 64, 128);
    tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    TCVT(a_ub, a_ub_half, pto::RoundMode::CAST_NONE);
    TMUL(a_ub, a_ub, coeff_ub);
    TMUL(a_ub, a_ub, msk_ub);
    TCVT(a_ub_half, a_ub, pto::RoundMode::CAST_NONE);
    tl::ascend_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128>(A_handle + ((cid * 16384) + (vid * 8192)), 67456, 0, 64, 128);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *K_handle, __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle, __gm__ uint8_t *workspace_handle, __gm__ uint8_t *A_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(K_handle),
     reinterpret_cast<__gm__ half *>(Beta_handle),
     reinterpret_cast<__gm__ float *>(G_handle),
     reinterpret_cast<__gm__ float *>(Msk_handle),
     reinterpret_cast<__gm__ half *>(workspace_handle),
     reinterpret_cast<__gm__ half *>(A_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *K_handle, uint8_t *Beta_handle, uint8_t *G_handle, uint8_t *Msk_handle, uint8_t *workspace_handle, uint8_t *A_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<4096, nullptr, stream>>>(K_handle, Beta_handle, G_handle, Msk_handle, workspace_handle, A_handle, fftsAddr);
}
