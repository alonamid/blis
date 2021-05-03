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

extern void bli_1v_hwacha_vf_init(void) __attribute__((visibility("protected")));
extern void bli_saxpbyv_unit_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_saxpbyv_stride_hwacha_vf_main(void) __attribute__((visibility("protected")));


void bli_saxpbyv_hwacha
     (
       conj_t           conjx,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       float*  restrict beta,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	if ( bli_zero_dim1( n ) ) return;

	if (n < HWACHA_MIN_DIM)
	{
#if defined(BLIS_CONFIG_GEMMINIHWACHA)
		bli_saxpbyv_gemminihwacha_ref
#else
		bli_saxpbyv_hwacha_ref
#endif
		(
			conjx,
			n,
			alpha,
			x, incx,
			beta,
			y, incy,
			cntx
		);
		return;
	}

	if ( *alpha == 0 )
	{
		if ( *beta == 0)
		{
			float* zero = 0; \
			/* Query the context for the kernel function pointer. */
			const num_t             dt     = BLIS_FLOAT;
			ssetv_ker_ft setv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_SETV_KER, cntx );

			setv_p
			(
			  BLIS_NO_CONJUGATE,
		 	  n,
			  zero,
			  y, incy,
			  cntx
			);
			return;
		}
		else if ( *beta == 1)
		{
			return;
		}
		else
		{
			/* If alpha is zero, scale by beta. */

			/* Query the context for the kernel function pointer. */
			const num_t              dt      = BLIS_FLOAT;
			sscalv_ker_ft scalv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_SCALV_KER, cntx );

			scalv_p
			(
			  BLIS_NO_CONJUGATE,
			  n,
			  beta,
			  y, incy,
			  cntx
			);
			return;
		}
	}
	else if ( *alpha == 1)
	{
		if ( *beta == 0)
		{
			/* If alpha is one and beta is zero, use copyv. */

			/* Query the context for the kernel function pointer. */
			const num_t              dt      = BLIS_FLOAT;
			scopyv_ker_ft copyv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_COPYV_KER, cntx );

			copyv_p
			(
			  conjx,
			  n,
			  x, incx,
			  y, incy,
			  cntx
			);
			return;
		}
		else if ( *beta == 1)
		{
			/* If alpha is one and beta is one, use addv. */

			/* Query the context for the kernel function pointer. */
			const num_t             dt     = BLIS_FLOAT;
			saddv_ker_ft addv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_ADDV_KER, cntx );

			addv_p
			(
			  conjx,
			  n,
			  x, incx,
			  y, incy,
			  cntx
			);
			return;
		}
		else
		{
			/* If alpha is one and beta is something else, use xpbyv. */

			/* Query the context for the kernel function pointer. */
			const num_t              dt      = BLIS_FLOAT;
			sxpbyv_ker_ft xpbyv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_XPBYV_KER, cntx );

			xpbyv_p
			(
			  conjx,
			  n,
			  x, incx,
			  beta,
			  y, incy,
			  cntx
			);
			return;
		}
	}
	else
	{
		if ( *beta == 0)
		{
			/* If alpha is something else and beta is zero, use scal2v. */

			/* Query the context for the kernel function pointer. */
			const num_t               dt       = BLIS_FLOAT;
			sscal2v_ker_ft scal2v_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_SCAL2V_KER, cntx );

			scal2v_p
			(
			  conjx,
			  n,
			  alpha,
			  x, incx,
			  y, incy,
			  cntx
			);
			return;
		}
		else if ( *beta == 1 )
		{
			/* If alpha is something else and beta is one, use axpyv. */

			/* Query the context for the kernel function pointer. */
			const num_t              dt      = BLIS_FLOAT;
			saxpyv_ker_ft axpyv_p = bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx );

			axpyv_p
			(
			  conjx,
			  n,
			  alpha,
			  x, incx,
			  y, incy,
			  cntx
			);
			return;
		}
	}

	__asm__ volatile ("vmcs vs1,  %0" : : "r" (*alpha));
	__asm__ volatile ("vmcs vs2,  %0" : : "r" (*beta));
	dim_t offset = 0;
	__asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 3, 0, 1)));
	int vlen_result;
	__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (n));
	vf(&bli_1v_hwacha_vf_init);
	if ( incx == 1 && incy == 1 )
	{
		for ( dim_t i = n; i > 0;)
		{
            		MEMTOUCH(y+offset, float, vlen_result);
            		MEMTOUCH(x+offset, float, vlen_result);
			__asm__ volatile ("vmca va0,  %0" : : "r" (y+offset));
			__asm__ volatile ("vmca va1,  %0" : : "r" (x+offset));
			vf(&bli_saxpbyv_unit_hwacha_vf_main);
			offset += vlen_result;
			i -= vlen_result;
	  		__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
		}
	}
	else
	{
		__asm__ volatile ("vmca va2,  %0" : : "r" (incy*sizeof(float)));
		__asm__ volatile ("vmca va3,  %0" : : "r" (incx*sizeof(float)));
		for ( dim_t i = n; i > 0;)
		{
            		MEMTOUCH(y+offset*incy, float, vlen_result*incy);
            		MEMTOUCH(x+offset*incx, float, vlen_result*incx);
			__asm__ volatile ("vmca va0,  %0" : : "r" (y+offset*incy));
			__asm__ volatile ("vmca va1,  %0" : : "r" (x+offset*incx));
			vf(&bli_saxpbyv_stride_hwacha_vf_main);
			offset += vlen_result;
			i -= vlen_result;
	  		__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
		}
	}
	__asm__ volatile ("fence" ::: "memory");
}
