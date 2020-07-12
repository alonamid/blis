/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


*/

#include "blis.h"
#include "include/gemmini.h"

void bli_sgemm_gemmini_small_os
     (
       dim_t               k0,
       float*    restrict alpha,
       float*    restrict a,
       float*    restrict b,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

	const num_t        dt     = BLIS_FLOAT;

	const dim_t        mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t        nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );

	const inc_t        packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx );
	const inc_t        packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );

	const inc_t        cs_a   = packmr;
	const inc_t        rs_b   = packnr;



       if (k0 == 0) 
       {
  
         //bli_obj_create
         //(
         //   num_t  dt,
         //   dim_t  mr,
         //   dim_t  nr,
         //   inc_t  rs_c0,
         //   inc_t  cs_c0,
         //   obj_t* obj
         //); 
         //bli_scalm( beta, c );
         bli_sscalm(
            BLIS_NO_CONJUGATE,
            0,
            BLIS_NONUNIT_DIAG,
            BLIS_DENSE,
            mr,
            nr,
            beta,
            c, rs_c0, cs_c0
         );

       } else {

        //printf("Starting Gemmini OS matmul\n");
        //printf("mr: %d, nr: %d, k0: %d\n", mr, nr, k0);
        //printf("a: %p, b: %p, c: %p\n", a, b, c);
        //printf("cs_a: %d, rs_b: %d, cs_c0: %d, rs_c0: %d\n", cs_a, rs_b, cs_c0, rs_c0);
        //printf("alpha: %f, beta: %f\n", *alpha, *beta);
/*
        printf("===Manual Result C====\n");
        for (int i=0; i<mr; i++) {
         for (int j=0; j<nr; j++) {
           float cval = *beta * *(c + i*rs_c0 + j*cs_c0);  
           for (int k=0; k<k0; k++) {
             cval += *alpha * *(a + k*cs_a + i) * *(b + k*rs_b + j); 
           }
           printf("%f ", cval);
          }
          printf("\n");
        }
*/
        //==============Gemmini Specific Code======================
        //we want mr=DIM and nr=DIM
        //k0 is unbounded
        //K_num_tiles is called K0 is gemmini.h

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

        const size_t dim_K_padded = (k0 / DIM + (k0 % DIM != 0)) * DIM;
        const size_t tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
        const size_t K_num_tiles = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);
        const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;
        const size_t padding_K = dim_K_padded - k0;

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k

        size_t I = mr / DIM + (mr % DIM != 0);
        size_t J = nr / DIM + (nr % DIM != 0);
        scale_t A_scale_factor = *alpha;
        scale_t B_scale_factor = 1.f;
        scale_acc_t D_scale_factor = *beta;
        size_t A_row_stride = cs_a;
        size_t B_row_stride = rs_b; 
        size_t D_row_stride = rs_c0;
        size_t C_row_stride = rs_c0;
        size_t C_col_stride = cs_c0;
        elem_t * A = a;
        elem_t * B = b;
        acc_t * D = c;
        elem_t * C = c;
        size_t pad_I = 0;
        size_t pad_J = 0;


        bool no_bias = (c == NULL) || (*beta == 0);
        if (no_bias) {
          D = (acc_t*) 1; // Dummy address which isn't NULL
        }


        //If C is column-major, we need to tranpose it
        static elem_t D_transpose[DIM][DIM] __attribute__ ((aligned (64)));
        static acc_t C_transpose[DIM][DIM] __attribute__ ((aligned (64)));
        bool C_column_major = false;
        if (C_row_stride == 1 && C_col_stride != 1) {
           C_column_major = true;
           C_row_stride = DIM;
           D_row_stride = DIM;
        }



        gemmini_extended_config_ex(OS, NO_ACTIVATION, 0, 0, 0, 1, true, false);
        gemmini_config_st(C_row_stride * sizeof(elem_t));


        //split K into tiles, since k0 is not bounded
        for (size_t K0 = 0; K0 < K_num_tiles*tile_K; K0+=tile_K) {

          const size_t K = K0 < (K_num_tiles-1)*tile_K ? tile_K : last_K;
          const size_t pad_K = K0 == (K_num_tiles-1)*tile_K ? padding_K : 0;


          // re-implement sp_tiled_matmul_os
          // without row-major assumption
          // and assuming A is transposed

          const uint32_t A_sp_addr_start = 0;
          const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
          const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
          const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

          //TODO: update this after Hasan updates the row stride to be able to load more than just DIM 
          //const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
          const int A_blocks = K <= 1 ? K : 1;
          const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
          const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;


          // Move-in D
          if (D != NULL && !no_bias) {
            const size_t D_stride = D_row_stride * sizeof(acc_t);
            gemmini_extended_config_ld(D_stride, D_scale_factor);
  
            for (size_t i = 0; i < I; i++) {
              for (size_t j = 0; j < J; j += D_blocks) {
                const size_t bias_row = i;
                const acc_t * D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j)*DIM;
        
                const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        
                size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
                const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
                const size_t rows = DIM - (i == I-1 ? pad_I : 0);

                //mini transpose if column-major
                if (C_column_major) {
                   gemmini_fence();
                   bli_scopys_mxn( mr,
                                nr,
                                (acc_t *)D + (bias_row * D_row_stride + j)*DIM,  rs_c0, cs_c0,
                                (elem_t*)D_transpose, DIM,  1 );
        
                   D_dram_addr = (acc_t *)D_transpose;
                }
                
                gemmini_extended_mvin(D_dram_addr, D_sp_addr_acc, cols, rows);
              }
            }
          }

          // Move-in B
          gemmini_extended_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor);
          for (size_t j = 0; j < J; j += B_blocks) {
            for (size_t k = 0; k < K; k++) {
              const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
              const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
              const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
              const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
              const size_t rows = DIM - (k == K-1 ? pad_K : 0);
              gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
            }
          }
        
          // Move-in A
          gemmini_extended_config_ld(A_row_stride * sizeof(elem_t), A_scale_factor);
          for (size_t i = 0; i < I; i++) {
            for (size_t k = 0; k < K; k += A_blocks) {
              //original
              //const elem_t * const A_dram_addr = A + (i * A_row_stride + k)*DIM;
              //const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
              //const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
              //const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
              //const size_t rows = DIM - (i == I-1 ? pad_I : 0);
              //new
              const elem_t * const A_dram_addr = A + (k * A_row_stride + i)*DIM;
              const uint32_t A_sp_addr = A_sp_addr_start + (k*I + i)*DIM;
              const size_t blocks = i + A_blocks <= I ? A_blocks : I-i;
              const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
              const size_t rows = DIM - (k == K-1 ? pad_K : 0);
              gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
            }
          }
        
          // Compute
          for (size_t i = 0; i < I; i++) {
            for (size_t j = 0; j < J; j++) {
              const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
        
              for (size_t k = 0; k < K; k++) {
        
                //const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
                const uint32_t A_sp_addr = A_sp_addr_start + (k*I + i)*DIM;
                const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
        
                uint32_t out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;
        
                // If we're not using a bias, then we want to overwrite what's in the
                // accumulator, rather than writing over it
                int no_bias_new_matrix = no_bias && D != NULL && k == K-1;
                if (no_bias_new_matrix) {
                  out_sp_addr &= ~(1 << (ADDR_LEN-2));
                }
        
                const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
                const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
                const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
                const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
                const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
                const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
        
                gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
        
                if (k == 0) { // First iteration
                  gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
                } else { // All other iterations
                  gemmini_extended_compute_accumulated(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
                }
              }
            }
          }
        
          gemmini_fence();

          // Move-out C
          if (C != NULL) {
            for (size_t i = 0; i < I; i++) {
              for (size_t j = 0; j < J; j++) {
                elem_t * const C_dram_addr = C_column_major ? (elem_t*)(&C_transpose) : C + (i*C_row_stride + j)*DIM;
                const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
 
                const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
                const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
        
                gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_cols, C_rows);

                //mini transpose if column-major
                if (C_column_major) {
                  gemmini_fence();
                  bli_scopys_mxn( mr,
                                nr,
                                (acc_t*)C_transpose,  DIM, 1,
                                C + (i*C_row_stride + j)*DIM, rs_c0,  cs_c0 );

                }
              }
            }
          }

          gemmini_fence();

        }
        

/*
         tiled_matmul(mr, nr, k0,
            a, b,
            c, c,
            cs_a, rs_b, cs_c0, cs_c0,
            *alpha, 1.f, *beta,
            NO_ACTIVATION, 0, 0, 0,
            mr/DIM + (mr % DIM != 0), nr/DIM + (nr % DIM != 0), k0/DIM + (k0 % DIM != 0),
            true, false,
            OS);
*/
/*
        printf("===Gemmini Result C====\n");
        for (int i=0; i<mr; i++) {
         for (int j=0; j<nr; j++) {
           printf("%f ", *(c + i*rs_c0 + j*cs_c0));
          }
          printf("\n");
        }
*/
      }

}
