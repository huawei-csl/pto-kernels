#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *K_handle, __gm__ half *V_handle, __gm__ half *Beta_handle, __gm__ float *G_handle, __gm__ half *A_handle, __gm__ half *workspace_a1_handle, __gm__ half *workspace_a2_handle, __gm__ half *W_handle, __gm__ half *U_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileUbDataND<half, 1, 128, 1, 128> beta_ub_half;
  TASSIGN(beta_ub_half, 0);
  tl::ascend_pto::TileUbDataND<half, 64, 128, 64, 128> a1_ub_half;
  TASSIGN(a1_ub_half, 256);
  tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> beta_ub;
  TASSIGN(beta_ub, 16640);
  tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> beta_r_ub;
  TASSIGN(beta_r_ub, 17152);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> beta_2d_ub;
  TASSIGN(beta_2d_ub, 17664);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 24576, 1, 24576> tmp_ub;
  TASSIGN(tmp_ub, 50432);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> a1_ub;
  TASSIGN(a1_ub, 75008);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> a2_ub;
  TASSIGN(a2_ub, 107776);
  tl::ascend_pto::TileUbDataND<half, 64, 128, 64, 128> a2_ub_half;
  TASSIGN(a2_ub_half, 140544);
  tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> g_ub;
  TASSIGN(g_ub, 156928);
  tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> g_r_ub;
  TASSIGN(g_r_ub, 157440);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> g_2d_ub;
  TASSIGN(g_2d_ub, 157952);
  tl::ascend_pto::TileMatL1<half, 128, 128, 128, 128> k_l1;
  TASSIGN(k_l1, 0);
  tl::ascend_pto::TileMatL1<half, 128, 128, 128, 128> v_l1;
  TASSIGN(v_l1, 32768);
  tl::ascend_pto::TileMatL1<half, 128, 128, 128, 128> a2_l1;
  TASSIGN(a2_l1, 65536);
  TileAcc<float, 128, 128, 128, 128> u_l0;
  TASSIGN(u_l0, 0);
  tl::ascend_pto::TileMatL1<half, 128, 128, 128, 128> a1_l1;
  TASSIGN(a1_l1, 98304);
  TileAcc<float, 128, 128, 128, 128> w_l0;
  TASSIGN(w_l0, 65536);
  auto vid = get_subblockid();
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 1, 128, 1, 1, 1, 524288, 1, 1, 128, pto::PadValue::Zero>(Beta_handle + (cid * 128), 0, 0, 1, 128);
    tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128, pto::PadValue::Zero>(A_handle + ((cid * 16384) + (vid * 8192)), 256, 0, 64, 128);
    TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TMOV(beta_r_ub, beta_ub);
    pipe_barrier(PIPE_V);
    TCOLEXPAND(beta_2d_ub, beta_r_ub);
    tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
    TMUL(a2_ub, a1_ub, beta_2d_ub);
    TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);
    tl::ascend_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128>(workspace_a2_handle + ((cid * 16384) + (vid * 8192)), 140544, 0, 64, 128);
    tl::ascend_pto::set_cross_flag<PIPE_MTE3>(2, 2);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 1, 128, 1, 1, 1, 524288, 1, 1, 128, pto::PadValue::Zero>(G_handle + (cid * 128), 156928, 0, 1, 128);
    tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
    TEXP(g_ub, g_ub);
    pipe_barrier(PIPE_V);
    TMUL(g_ub, g_ub, beta_ub);
    pipe_barrier(PIPE_V);
    TMOV(g_r_ub, g_ub);
    pipe_barrier(PIPE_V);
    TCOLEXPAND(g_2d_ub, g_r_ub);
    TMUL(a1_ub, a1_ub, g_2d_ub);
    TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);
    tl::ascend_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> (0);
    tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 67108864, 33554432, 2097152, 128, 1, 64, 128>(workspace_a1_handle + ((cid * 16384) + (vid * 8192)), 256, 0, 64, 128);
    tl::ascend_pto::set_cross_flag<PIPE_MTE3>(1, 2);
#endif
#if defined(__DAV_C220_CUBE__)
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(K_handle + (cid * 16384), 0, 0, 128, 128);
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(V_handle + (cid * 16384), 32768, 0, 128, 128);
    tl::ascend_pto::wait_cross_flag(2);
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(workspace_a2_handle + (cid * 16384), 65536, 0, 128, 128);
    tl::ascend_pto::gemm_v0<half, float, 128, 128, 128, 128, 128, 128, 128, false, false>(a2_l1, v_l1, u_l0, (bool)1);
    tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(U_handle + (cid * 16384), 0, 0, 128, 128);
    tl::ascend_pto::wait_cross_flag(1);
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(workspace_a1_handle + (cid * 16384), 98304, 0, 128, 128);
    tl::ascend_pto::gemm_v0<half, float, 128, 128, 128, 128, 128, 128, 128, false, false>(a1_l1, k_l1, w_l0, (bool)1);
    tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 67108864, 33554432, 2097152, 128, 1, 128, 128>(W_handle + (cid * 16384), 65536, 0, 128, 128);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle, __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle, __gm__ uint8_t *A_handle, __gm__ uint8_t *workspace_a1_handle, __gm__ uint8_t *workspace_a2_handle, __gm__ uint8_t *W_handle, __gm__ uint8_t *U_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(K_handle),
     reinterpret_cast<__gm__ half *>(V_handle),
     reinterpret_cast<__gm__ half *>(Beta_handle),
     reinterpret_cast<__gm__ float *>(G_handle),
     reinterpret_cast<__gm__ half *>(A_handle),
     reinterpret_cast<__gm__ half *>(workspace_a1_handle),
     reinterpret_cast<__gm__ half *>(workspace_a2_handle),
     reinterpret_cast<__gm__ half *>(W_handle),
     reinterpret_cast<__gm__ half *>(U_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *K_handle, uint8_t *V_handle, uint8_t *Beta_handle, uint8_t *G_handle, uint8_t *A_handle, uint8_t *workspace_a1_handle, uint8_t *workspace_a2_handle, uint8_t *W_handle, uint8_t *U_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<4096, nullptr, stream>>>(K_handle, V_handle, Beta_handle, G_handle, A_handle, workspace_a1_handle, workspace_a2_handle, W_handle, U_handle, fftsAddr);
}
