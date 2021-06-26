/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2017 - 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of copyright holder(s) nor the names
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
#include "blix.h"
#include "include/gemmini.h"

#define MEMTOUCH(iaddr, type, bound) ({                           \
      volatile type* addr = iaddr;                                \
      volatile type t;                                            \
      t = (addr)[0];                                              \
      (addr)[0] = t;                                              \
      volatile type* tf = (type*) (((((uintptr_t) (addr)) >> 12) + 1) << 12);    \
      for (; tf - (addr) < bound; tf += (1 << 12) / sizeof(type)) {     \
        t = tf[0];                                                      \
        tf[0] = t;                                                      \
      }                                                                 \
      __asm__ __volatile__ ("fence" ::: "memory"); \
    })


// Given the current architecture of BLIS sandboxes, bli_gemmnat() is the
// entry point to any sandbox implementation.

// NOTE: We must keep this function named bli_gemmnat() since this is the BLIS
// API function for which we are providing an alternative implementation via
// the sandbox.

void bli_gemmnat
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	// Obtain a valid (native) context from the gks if necessary.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();


	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } 

	const num_t    dt        = bli_obj_dt( c );

	const bool     packa     = bli_rntm_pack_a( rntm );
	const bool     packb     = bli_rntm_pack_b( rntm );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );
	      dim_t    k;

	void* restrict buf_a = bli_obj_buffer_at_off( a );
	      inc_t    rs_a;
	      inc_t    cs_a;

	void* restrict buf_b = bli_obj_buffer_at_off( b );
	      inc_t    rs_b;
	      inc_t    cs_b;


	bool atrans = false;
	bool btrans = false;

	if ( bli_obj_has_notrans( a ) )
	{
		k     = bli_obj_width( a );

		rs_a  = bli_obj_row_stride( a );
		cs_a  = bli_obj_col_stride( a );
	}
	else // if ( bli_obj_has_trans( a ) )
	{
		// Assign the variables with an implicit transposition.
		k     = bli_obj_length( a );

		rs_a  = bli_obj_col_stride( a );
		cs_a  = bli_obj_row_stride( a );
	}

	if ( bli_obj_has_notrans( b ) )
	{
		rs_b  = bli_obj_row_stride( b );
		cs_b  = bli_obj_col_stride( b );
	}
	else // if ( bli_obj_has_trans( b ) )
	{
		// Assign the variables with an implicit transposition.
		rs_b  = bli_obj_col_stride( b );
		cs_b  = bli_obj_row_stride( b );
	}

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

	/* tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
	weightA,
        enum tiled_matmul_type_t tiled_matmul_type) */

	if (k == 0)
	{
		bli_scalm(beta, c);
		return;
	}

	if (dt == BLIS_FLOAT && cs_a == 1 && cs_b == 1 && cs_c == 1)
	{
		MEMTOUCH(buf_a, float, m*rs_a);
		MEMTOUCH(buf_b, float, k*rs_b);
		MEMTOUCH(buf_c, float, m*rs_c);
		if ( atrans && btrans )
	        {
			tiled_matmul_auto(m, n, k,
					(float*)buf_b, (float*)buf_a,
					(float*)buf_c, (float*)buf_c,
					rs_a, rs_b, rs_c, rs_c,
					*((float*)buf_alpha), MVIN_SCALE_IDENTITY, *((float*)buf_beta),
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					true, false,
					1,
					WS);
		}
		else
		{
			tiled_matmul_auto(m, n, k,
					(float*)buf_a, (float*)buf_b,
					(float*)buf_c, (float*)buf_c,
					rs_a, rs_b, rs_c, rs_c,
					*((float*)buf_alpha), MVIN_SCALE_IDENTITY, *((float*)buf_beta),
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					atrans, btrans,
					true, false,
					1,
					WS);
		}
	}
	else if (dt == BLIS_FLOAT && rs_a == 1 && rs_b == 1 && rs_c == 1)
	{
		MEMTOUCH(buf_a, float, k*cs_a);
		MEMTOUCH(buf_b, float, n*cs_b);
		MEMTOUCH(buf_c, float, n*cs_c);
		if ( atrans && btrans )
	        {
			tiled_matmul_auto(n, m, k,
					(float*)buf_a, (float*)buf_b,
					(float*)buf_c, (float*)buf_c,
					cs_a, cs_b, cs_c, cs_c,
					*((float*)buf_alpha), MVIN_SCALE_IDENTITY, *((float*)buf_beta),
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					true, false,
					1,
					WS);
		} 
		else
		{
			tiled_matmul_auto(n, m, k,
					(float*)buf_b, (float*)buf_a,
					(float*)buf_c, (float*)buf_c,
					cs_b, cs_a, cs_c, cs_c,
					*((float*)buf_alpha), MVIN_SCALE_IDENTITY, *((float*)buf_beta),
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					btrans, atrans,
					true, false,
					1,
					WS);
		}
	} else {
                bli_gemm_front
                (
                  alpha,
                  a,
                  b,
                  beta,
                  c,
                  cntx,
                  rntm,
                  NULL
                );
/*
		blx_gemm_ref_var2( BLIS_NO_TRANSPOSE,
					alpha, a, b, beta, c,
					BLIS_XXX, cntx, rntm, NULL );
*/
	}
}

