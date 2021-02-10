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


void bli_sgemm_gemmini_fsm_ws
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

	const bool       elem_out = bli_cntx_lowprec_elem_out(cntx);

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

        //printf("Starting Gemmini WS matmul\n");
        //printf("mr: %d, nr: %d, k0: %d\n", mr, nr, k0);
        //printf("a: %p, b: %p, c: %p\n", a, b, c);
        //printf("cs_a: %d, rs_b: %d, cs_c0: %d, rs_c0: %d\n", cs_a, rs_b, cs_c0, rs_c0);
        //printf("alpha: %f, beta: %f\n", *alpha, *beta);

/*
        printf("===GEMM Microkernel A panel====\n");
        for (int i=0; i<mr; i++) {
           for (int k=0; k<k0; k++) {
             float a_f;
             bli_tofloat(*((elem_t*)a + k*cs_a + i), a_f);
             printf("%f ", a_f);
          }
          printf("\n");
        }
        printf("===GEMM Microkernel B panel====\n");
        for (int i=0; i<nr; i++) {
           for (int k=0; k<k0; k++) {
             float b_f;
             bli_tofloat(*((elem_t*)b + k*rs_b + i), b_f);
             printf("%f ", b_f);
          }
          printf("\n");
        }

        printf("===GEMM Microkernel C Bias====\n");
        for (int i=0; i<mr; i++) {
         for (int j=0; j<nr; j++) {
	   if (elem_out)
	   {
             float c_f;
             bli_tofloat(*((elem_t*)c + i*rs_c0 + j*cs_c0), c_f);
             printf("%f ", c_f);
           } else {
             printf("%f ", *(c + i*rs_c0 + j*cs_c0));
           }
         }
          printf("\n");
        }

        printf("===GEMM Microkernel Manual Result C====\n");
        for (int i=0; i<mr; i++) {
         for (int j=0; j<nr; j++) {
           float cval = *beta * *(c + i*rs_c0 + j*cs_c0);
           for (int k=0; k<k0; k++) {
             float a_f;
             float b_f;
             bli_tofloat(*((elem_t*)a + k*cs_a + i), a_f);
             bli_tofloat(*((elem_t*)b + k*rs_b + j), b_f);
             cval += *alpha * a_f * b_f;
             //cval += *alpha * *((elem_t*)a + k*cs_a + i) * *((elem_t*)b + k*rs_b + j);
           }
           printf("%f ", cval);
          }
          printf("\n");
        }
*/

        //==============Gemmini Specific Code======================
        //we want mr=sqrt(accumulators) and nr=sqrt(accumulators)         //k0 is unbounded
        //K_num_tiles is called K0 is gemmini.h

//single buffering
/*
#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)
*/
//double buffering use half the memory resources
#define partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc ((ACC_ROWS / 2) / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)


        const size_t dim_K_padded = (k0 / DIM + (k0 % DIM != 0)) * DIM;
        const size_t tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
        const size_t padding_K = dim_K_padded - k0;
        const size_t K_num_tiles = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);
        const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;
        const size_t last_K0 = (K_num_tiles - 1)*tile_K;

        const size_t dim_I_padded = (mr / DIM + (mr % DIM != 0)) * DIM;
        const size_t tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
        const size_t padding_I = dim_I_padded - mr;

        const size_t dim_J_padded = (nr / DIM + (nr % DIM != 0)) * DIM;
        const size_t tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
        const size_t padding_J = dim_J_padded - nr;


        scale_t A_scale_factor = *alpha;
        scale_t B_scale_factor = 1.f;
        scale_acc_t D_scale_factor = *beta;
        size_t A_row_stride = cs_a;
        size_t B_row_stride = rs_b;
        size_t D_row_stride = rs_c0;
        size_t C_row_stride = rs_c0;
        size_t C_col_stride = cs_c0;
        elem_t * A = (elem_t*)a;
        elem_t * B = (elem_t*)b;
        acc_t * D = c;
        acc_t * C = c;
        const size_t I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
        const size_t pad_I = padding_I;
        const size_t J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
        const size_t pad_J = padding_J;

        bool no_bias = (c == NULL) || (*beta == 0);

        //If C is column-major, we will need to tranpose it
        static acc_t D_transpose[mats_in_acc*DIM*DIM] __attribute__ ((aligned (64)));
        acc_t* C_transpose = D_transpose;
        bool C_column_major = false;
        if (C_row_stride == 1 && C_col_stride != 1) {
           C_column_major = true;
           C_row_stride = mr;
           D_row_stride = mr;
        }

        const size_t C_stride = elem_out ? C_row_stride * sizeof(elem_t) :  C_row_stride * sizeof(acc_t);
        const size_t D_stride = C_stride;
        acc_t * const C_dram_addr = C_column_major ? (acc_t*)(C_transpose) : C;
        const acc_t * D_dram_addr = D;

        //mini transpose if column-major
        if (C_column_major) {
          gemmini_fence();

          const size_t D_cols = nr - (J == nr ? pad_J : 0);
          const size_t D_rows = mr - (I == mr ? pad_I : 0);
          //bli_scopys_mxn( rows,
          //             cols,
          //             (acc_t *)D + i*DIM*rs_c0 + j*DIM*cs_c0,  rs_c0, cs_c0,
          //             (elem_t*)D_transpose, MAX_BLOCK_LEN_ACC*DIM,  1 );
          if (elem_out)
          {
#ifdef BLIS_ENABLE_CR_CASES
            if ( cs_c0 == 1 )
            {
              for ( dim_t ii = 0; ii < D_rows; ++ii )
              for ( dim_t jj = 0; jj < D_cols; ++jj )
                *((elem_t*)D_transpose + ii*mr + jj) = *((elem_t *)D + ii*rs_c0 + jj);
            }
            else
#endif
            {
              for ( dim_t jj = 0; jj < D_cols; ++jj )
              for ( dim_t ii = 0; ii < D_rows; ++ii )
                *((elem_t*)D_transpose + ii*mr + jj) = *((elem_t *)D + ii*rs_c0 + jj*cs_c0);
            }

          } else {
#ifdef BLIS_ENABLE_CR_CASES
            if ( cs_c0 == 1 )
            {
              for ( dim_t ii = 0; ii < D_rows; ++ii )
              for ( dim_t jj = 0; jj < D_cols; ++jj )
                *((acc_t*)D_transpose + ii*mr + jj) = *((acc_t *)D + ii*rs_c0 + jj);
            }
            else
#endif
            {
              for ( dim_t jj = 0; jj < D_cols; ++jj )
              for ( dim_t ii = 0; ii < D_rows; ++ii )
                *((acc_t*)D_transpose + ii*mr + jj) = *((acc_t *)D + ii*rs_c0 + jj*cs_c0);
            }

          }

          D_dram_addr = (acc_t *)(D_transpose);

        }

        // configure gemmini only once to enable double buffering
        if (bli_cntx_lowprec_start(cntx) || D_stride != bli_auxinfo_lowprec_prev_stride(data) || D_scale_factor != bli_auxinfo_lowprec_prev_scale(data)) {
          gemmini_extended_config_ex(WS, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 0, 1, true, false)
          gemmini_config_st(C_stride);
          gemmini_extended3_config_ld(A_row_stride * sizeof(elem_t), A_scale_factor, elem_out, 0);
          gemmini_extended3_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor, elem_out, 1);
          gemmini_extended3_config_ld(D_stride, D_scale_factor, elem_out, 2);
          bli_cntx_set_lowprec_start(cntx, 0);
          bli_auxinfo_set_lowprec_prev_stride(data, D_stride);
          bli_auxinfo_set_lowprec_prev_scale(data, D_scale_factor);
        }

        // tile based on the scratchpad size
        for (size_t K0 = 0; K0 < K_num_tiles*tile_K; K0 += tile_K)
        {
          bool last_tile = K0 == last_K0;
          const elem_t * const B_dram_addr = B + (K0*DIM*B_row_stride);
          const elem_t * const A_dram_addr = A + (K0*DIM*A_row_stride);
          const size_t K = last_tile ? last_K : tile_K;
          const size_t pad_K = last_tile ? padding_K : 0;

          // Call gemmini matmul FSM
          gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, 
                          A_dram_addr, B_dram_addr, 
                          (K0 == 0 && !no_bias) ? D_dram_addr : NULL, 
                          last_tile ? C_dram_addr : NULL,
                          A_row_stride, B_row_stride, D_row_stride, C_row_stride,
                          true, false,
                          !elem_out, elem_out, !(K0 == 0) || !no_bias);

        }

        //fence if this is a gemmtrsm kernel (elem_out)
        if (elem_out) gemmini_fence();

        //mini transpose if column-major
        if (C_column_major) {
          gemmini_fence();

          const size_t C_cols = nr - (J == nr ? pad_J : 0);
          const size_t C_rows = nr - (I == nr ? pad_I : 0);
          //bli_scopys_mxn( C_rows,
          //              C_cols,
          //              (acc_t*)(C_transpose),  DIM, 1,
          //              C + i*DIM*rs_c0 + j*DIM*cs_c0,  rs_c0, cs_c0 );
          if (elem_out)
          {
#ifdef BLIS_ENABLE_CR_CASES
            if ( cs_c0 == 1 )
            {
              for ( dim_t ii = 0; ii < C_rows; ++ii )
              for ( dim_t jj = 0; jj < C_cols; ++jj )
                *((elem_t*)C + ii*rs_c0 + jj) = *((elem_t*)(C_transpose) + ii*nr + jj);
            }
            else
#endif
            {
              for ( dim_t jj = 0; jj < C_cols; ++jj )
              for ( dim_t ii = 0; ii < C_rows; ++ii )
                *((elem_t*)C + ii*rs_c0 + jj*cs_c0) = *((elem_t*)(C_transpose) + ii*nr + jj);
            }
          } else {
#ifdef BLIS_ENABLE_CR_CASES
            if ( cs_c0 == 1 )
            {
              for ( dim_t ii = 0; ii < C_rows; ++ii )
              for ( dim_t jj = 0; jj < C_cols; ++jj )
                *(C + ii*rs_c0 + jj) = *((acc_t*)(C_transpose) + ii*nr + jj);
            }
            else
#endif
            {
              for ( dim_t jj = 0; jj < C_cols; ++jj )
              for ( dim_t ii = 0; ii < C_rows; ++ii )
                *(C + ii*rs_c0 + jj*cs_c0) = *((acc_t*)(C_transpose) + ii*nr + jj);
            }
          }
        }

/*
        gemmini_fence();
        printf("===GEMM Microkernel Result C====\n");
        for (int i=0; i<mr; i++) {
         for (int j=0; j<nr; j++) {
	   if (elem_out)
	   {
             float c_f;
             bli_tofloat(*((elem_t*)c + i*rs_c0 + j*cs_c0), c_f);
             printf("%f ", c_f);
           } else {
             printf("%f ", *(c + i*rs_c0 + j*cs_c0));
           }
         }
          printf("\n");
        }
*/

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k

      }

}
