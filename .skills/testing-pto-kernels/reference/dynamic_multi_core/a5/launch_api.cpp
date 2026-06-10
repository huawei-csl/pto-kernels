#include <cstdint>

void LaunchMatmulAddC2V(uint32_t block_dim, uint8_t *A, uint8_t *B, uint8_t *C, uint8_t *D, int64_t batch,
                        void *stream);
void LaunchAddMatmulV2C(uint32_t block_dim, uint8_t *A, uint8_t *B, uint8_t *C, uint8_t *D, int64_t batch,
                        void *stream);

extern "C" void cv_matmul_add_c2v(uint32_t block_dim, uint8_t *A, uint8_t *B, uint8_t *C, uint8_t *D, int64_t batch,
                                   void *stream)
{
    LaunchMatmulAddC2V(block_dim, A, B, C, D, batch, stream);
}

extern "C" void cv_add_matmul_v2c(uint32_t block_dim, uint8_t *A, uint8_t *B, uint8_t *C, uint8_t *D, int64_t batch,
                                   void *stream)
{
    LaunchAddMatmulV2C(block_dim, A, B, C, D, batch, stream);
}

