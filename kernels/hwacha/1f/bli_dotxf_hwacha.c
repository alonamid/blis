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

extern void bli_sdotxf_hwacha_vf_init(void) __attribute__((visibility("protected")));
extern void bli_sdotxf_hwacha_vf_init_beta(void) __attribute__((visibility("protected")));
extern void bli_sdotxf_hwacha_vf_early_end(void) __attribute__((visibility("protected")));
extern void bli_sdotxf_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_sdotxf_hwacha_vf_alpha(void) __attribute__((visibility("protected")));

void bli_sdotxf_hwacha
     (
       conj_t           conjat,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       float*  restrict alpha,
       float*  restrict a, inc_t inca, inc_t lda,
       float*  restrict x, inc_t incx,
       float*  restrict beta,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	if ( inca == 1 && incx == 1 && incy == 1 && b_n >= HWACHA_MIN_DIM )
	{
	  int vlen_result;      
	  dim_t offset = 0;
	  __asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 4, 0, 1)));
	  __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (b_n));
	  for ( dim_t i = b_n; i > 0;) {

            	MEMTOUCH(y+offset, float, vlen_result);
                __asm__ volatile ("vmca va0,  %0" : : "r" (y+offset));
		/* If beta is zero, clear y. Otherwise, scale by beta. */
		if (  *beta == 0 )
		{
			vf(&bli_sdotxf_hwacha_vf_init);
		}
		else
		{
                	__asm__ volatile ("vmcs vs1,  %0" : : "r" (*beta));
			vf(&bli_sdotxf_hwacha_vf_init_beta);
		}

		/* If the vectors are empty or if alpha is zero, return early. */
		if ( bli_zero_dim1( m ) || ( *alpha == 0 ) ) 
		{
			vf(&bli_sdotxf_hwacha_vf_early_end);
			offset += vlen_result;
			i -= vlen_result;
	  		__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
			continue;
		}

		for ( dim_t p = 0; p < m; ++p )
		{
            		MEMTOUCH(a+p+offset*lda, float, vlen_result*lda);
                	__asm__ volatile ("vmca va1,  %0" : : "r" (a + p + offset*lda));
                	__asm__ volatile ("vmca va2,  %0" : : "r" (lda*sizeof(float)));
                	__asm__ volatile ("vmcs vs2,  %0" : : "r" (x[p]));
			vf(&bli_sdotxf_hwacha_vf_main);
		}
                __asm__ volatile ("vmcs vs3,  %0" : : "r" (*alpha));
		vf(&bli_sdotxf_hwacha_vf_alpha);
		offset += vlen_result;
		i -= vlen_result;
	  	__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
	  }
	  __asm__ volatile ("fence" ::: "memory");
	}
	else
	{
		/* Query the context for the kernel function pointer. */
		const num_t              dt     = BLIS_FLOAT;
		sdotxv_ker_ft kfp_dv
		= 
		bli_cntx_get_l1v_ker_dt( dt, BLIS_DOTXV_KER, cntx );

		for ( dim_t i = 0; i < b_n; ++i )
		{
			float* restrict a1   = a + (0  )*inca + (i  )*lda;
			float* restrict x1   = x + (0  )*incx;
			float* restrict psi1 = y + (i  )*incy;

			kfp_dv
			(
			  conjat,
			  conjx,
			  m,
			  alpha,
			  a1, inca,
			  x1, incx,
			  beta,
			  psi1,
			  cntx
			);
		}
	}
}
