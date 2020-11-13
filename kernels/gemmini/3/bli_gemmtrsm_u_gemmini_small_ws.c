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

void bli_sgemmtrsm_u_gemmini_small_ws
     (
       dim_t               k,
       float*    restrict alpha,
       float*    restrict a12,
       float*    restrict a11,
       float*    restrict b21,
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


        /* b11 = alpha * b11 - a12 * b21; */

	bli_cntx_set_lowprec_elem_out(cntx, 1);
        bli_sgemm_gemmini_small_ws
        (
          k,
          minus_one,
          a12,
          b21,
          alpha,
          b11, rs_b, cs_b,
          data,
          cntx
        );
	bli_cntx_set_lowprec_elem_out(cntx, 0);

        /* b11 = inv(a11) * b11;
           c11 = b11; */
        bli_strsm_u_gemmini_small
        (
          a11,
          b11,
          c11, rs_c, cs_c,
          data,
          cntx
        );

}
