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

void bli_sgemmtrsm_l_gemmini_fsm_ws
     (
       dim_t               k,
       float*    restrict alpha,
       float*    restrict a10,
       float*    restrict a11,
       float*    restrict b01,
       float*    restrict b11,
       float*    restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

	const num_t        dt     = BLIS_FLOAT;

	const inc_t        packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );

	const inc_t        rs_b   = packnr;
        const inc_t        cs_b   = 1;


        float* restrict minus_one = bli_sm1;
/*
	const inc_t        packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx );
        printf("===GEMMTRSM Microkernel A10 panel====\n");
        for (int i=0; i<packmr; i++) {
           for (int kk=0; kk<k; kk++) {
             float a_f;
             bli_tofloat(*((elem_t*)a10 + kk*packmr + i), a_f);
             printf("%f ", a_f);
          }
          printf("\n");
        }
        printf("===GEMMTRSM Microkernel A11 panel====\n");
        for (int i=0; i<packmr; i++) {
           for (int kk=0; kk<k; kk++) {
             float a_f;
             bli_tofloat(*((elem_t*)a11 + kk*packmr + i), a_f);
             printf("%f ", a_f);
          }
          printf("\n");
        }
        printf("===GEMMTRSM Microkernel B01 panel====\n");
        for (int i=0; i<packnr; i++) {
           for (int kk=0; kk<k; kk++) {
             float b_f;
             bli_tofloat(*((elem_t*)b01 + kk*packnr + i), b_f);
             printf("%f ", b_f);
          }
          printf("\n");
        }
        printf("===GEMMTRSM Microkernel B11 panel====\n");
        for (int i=0; i<packnr; i++) {
           for (int kk=0; kk<k; kk++) {
             float b_f;
             bli_tofloat(*((elem_t*)b11 + kk*packnr + i), b_f);
             printf("%f ", b_f);
          }
          printf("\n");
        }
       printf("===GEMMTRSM Microkernel C11 panel====\n");
        for (int ii=0; ii<packmr; ii++) {
         for (int jj=0; jj<packnr; jj++) {
           printf("%f ", *(c11 + ii*rs_c + jj*cs_c));
         }
          printf("\n");
        }
*/

        /* b11 = alpha * b11 - a10 * b01; */

	bli_cntx_set_lowprec_elem_out(cntx, 1);
        bli_sgemm_gemmini_fsm_ws
        (
          k,
          minus_one,
          a10,
          b01,
          alpha,
          b11, rs_b, cs_b,
          data,
          cntx
        );
	bli_cntx_set_lowprec_elem_out(cntx, 0);


        /* b11 = inv(a11) * b11;
           c11 = b11; */

        bli_strsm_l_gemmini_small
        (
          a11,
          b11,
          c11, rs_c, cs_c,
          data,
          cntx
        );

/*
       printf("===GEMMTRSM Microkernel Gemmini Result C====\n");
        for (int ii=0; ii<mr; ii++) {
         for (int jj=0; jj<nr; jj++) {
           printf("%f ", *(c11 + ii*rs_c + jj*cs_c));
         }
          printf("\n");
        }
*/
}
