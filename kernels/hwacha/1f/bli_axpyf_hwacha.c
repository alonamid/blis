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

extern void bli_saxpyf_hwacha_vf_init(void) __attribute__((visibility("protected")));
extern void bli_saxpyf_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_saxpyf_hwacha_vf_end(void) __attribute__((visibility("protected")));

void bli_saxpyf_hwacha
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       float*  restrict alpha,
       float*  restrict a, inc_t inca, inc_t lda,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	if ( bli_zero_dim1( m ) ) return;

	if ( inca == 1 && incx == 1 && incy == 1 && m >= HWACHA_MIN_DIM )
	{
	  int vlen_result;      
	  __asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 3, 0, 1)));
	  __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (m));
	  for ( dim_t i = 0; i < m;) {

                __asm__ volatile ("vmca va0,  %0" : : "r" (y+i));
		vf(&bli_saxpyf_hwacha_vf_init);

		for ( dim_t j = 0; j < b_n; ++j )
		{
                	__asm__ volatile ("vmca va1,  %0" : : "r" (a + i + j*lda));
                	__asm__ volatile ("vmcs vs1,  %0" : : "r" (x[j]));
			vf(&bli_saxpyf_hwacha_vf_main);
		}
                __asm__ volatile ("vmcs vs2,  %0" : : "r" (*alpha));
		vf(&bli_saxpyf_hwacha_vf_end);
	  	__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (m-i));
		i += vlen_result;
	  }
	  __asm__ volatile ("fence" ::: "memory");
	}
	else
	{
		/* Query the context for the kernel function pointer. */
		const num_t              dt     = BLIS_FLOAT;
		saxpyv_ker_ft kfp_av
		=
		bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx );

		for ( dim_t i = 0; i < b_n; ++i )
		{
			float* restrict a1   = a + (0  )*inca + (i  )*lda;
			float* restrict chi1 = x + (i  )*incx;
			float* restrict y1   = y + (0  )*incy;

			float alpha_chi1;

			bli_scopycjs( conjx, *chi1, alpha_chi1 );
			bli_sscals( *alpha, alpha_chi1 );

			kfp_av
			(
			  conja,
			  m,
			  &alpha_chi1,
			  a1, inca,
			  y1, incy,
			  cntx
			);
		}
	}
}
