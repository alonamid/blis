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
extern void bli_1v_hwacha_vf_hinit(void) __attribute__((visibility("protected")));
extern void bli_sscal2v_unit_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_sscal2v_stride_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_hscal2v_unit_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_hscal2v_stride_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_scopyv_unit_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_scopyv_stride_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_hcopyv_unit_hwacha_vf_main(void) __attribute__((visibility("protected")));
extern void bli_hcopyv_stride_hwacha_vf_main(void) __attribute__((visibility("protected")));

void bli_sscal2v_hwacha
     (
       conj_t           conjalpha,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	if ( bli_zero_dim1( n ) ) return;

	/* If alpha is zero, use setv. */
	if ( bli_seq0( *alpha ) )
	{
		float* zero = bli_s0;
	
		bli_ssetv_hwacha
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  zero,
		  y, incy,
		  cntx
		);
		return;
	}

	dim_t offset = 0;
	if ( bli_seq1( *alpha ) )
	{
		if (bli_cntx_lowprec_in_use(cntx))
		{
			elem_t* restrict x_elem = (elem_t*)x;
			elem_t* restrict y_elem = (elem_t*)y;
			__asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 0, 1, 1)));
			int vlen_result;
			__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (n));

			vf(&bli_1v_hwacha_vf_init);

			if ( incx == 1 )
			{
				for ( dim_t i = n; i > 0;)
				{
            				MEMTOUCH(y_elem+offset, elem_t, vlen_result);
            				MEMTOUCH(x_elem+offset, elem_t, vlen_result);
					__asm__ volatile ("vmca va0,  %0" : : "r" (x_elem+offset));
					__asm__ volatile ("vmca va1,  %0" : : "r" (y_elem+offset));
					vf(&bli_hcopyv_unit_hwacha_vf_main);
					offset += vlen_result;
					i -= vlen_result;
		  			__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
				}
			}
			else
			{
				__asm__ volatile ("vmca va2,  %0" : : "r" (incx*sizeof(elem_t)));
				__asm__ volatile ("vmca va3,  %0" : : "r" (incy*sizeof(elem_t)));
				for ( dim_t i = n; i > 0;)
				{
            				MEMTOUCH(y_elem+offset*incy, elem_t, vlen_result*incy);
            				MEMTOUCH(x_elem+offset*incx, elem_t, vlen_result*incx);
					__asm__ volatile ("vmca va0,  %0" : : "r" (x_elem+offset*incx));
					__asm__ volatile ("vmca va1,  %0" : : "r" (y_elem+offset*incy));
					vf(&bli_hcopyv_stride_hwacha_vf_main);
					offset += vlen_result;
					i -= vlen_result;
	  				__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
				}
			}
		} else {

			__asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 1, 0, 1)));
			int vlen_result;
			__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (n));

			__asm__ volatile ("vmcs vs1,  %0" : : "r" (*alpha));
			vf(&bli_1v_hwacha_vf_init);

			if ( incx == 1 )
			{
				for ( dim_t i = n; i > 0;)
				{
            				MEMTOUCH(y+offset, float, vlen_result);
            				MEMTOUCH(x+offset, float, vlen_result);
					__asm__ volatile ("vmca va0,  %0" : : "r" (x+offset));
					__asm__ volatile ("vmca va1,  %0" : : "r" (y+offset));
					vf(&bli_scopyv_unit_hwacha_vf_main);
					offset += vlen_result;
					i -= vlen_result;
	  				__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
				}
			}
			else
			{
				__asm__ volatile ("vmca va2,  %0" : : "r" (incx*sizeof(float)));
				__asm__ volatile ("vmca va3,  %0" : : "r" (incy*sizeof(float)));
				for ( dim_t i = n; i > 0;)
				{
            				MEMTOUCH(y+offset*incy, float, vlen_result*incy);
            				MEMTOUCH(x+offset*incx, float, vlen_result*incx);
					__asm__ volatile ("vmca va0,  %0" : : "r" (x+offset*incx));
					__asm__ volatile ("vmca va1,  %0" : : "r" (y+offset*incy));
					vf(&bli_scopyv_stride_hwacha_vf_main);
					offset += vlen_result;
					i -= vlen_result;
	  				__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
				}
			}
		}
		return;
	}

	if (bli_cntx_lowprec_in_use(cntx))
	{
		elem_t* restrict x_elem = (elem_t*)x;
		elem_t* restrict y_elem = (elem_t*)y;
		__asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 0, 1, 1)));
		int vlen_result;
		__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (n));

		__asm__ volatile ("vmcs vs1,  %0" : : "r" (*alpha));
		vf(&bli_1v_hwacha_vf_hinit);

		if ( incx == 1 )
		{
			for ( dim_t i = n; i > 0;)
			{
            			MEMTOUCH(y_elem+offset, elem_t, vlen_result);
            			MEMTOUCH(x_elem+offset, elem_t, vlen_result);
				__asm__ volatile ("vmca va0,  %0" : : "r" (x_elem+offset));
				__asm__ volatile ("vmca va1,  %0" : : "r" (y_elem+offset));
				vf(&bli_hscal2v_unit_hwacha_vf_main);
				offset += vlen_result;
				i -= vlen_result;
	  			__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
			}
		}
		else
		{
			__asm__ volatile ("vmca va2,  %0" : : "r" (incx*sizeof(elem_t)));
			__asm__ volatile ("vmca va3,  %0" : : "r" (incy*sizeof(elem_t)));
			for ( dim_t i = n; i > 0;)
			{
            			MEMTOUCH(y_elem+offset*incy, elem_t, vlen_result*incy);
            			MEMTOUCH(x_elem+offset*incx, elem_t, vlen_result*incx);
				__asm__ volatile ("vmca va0,  %0" : : "r" (x_elem+offset*incx));
				__asm__ volatile ("vmca va1,  %0" : : "r" (y_elem+offset*incy));
				vf(&bli_hscal2v_stride_hwacha_vf_main);
				offset += vlen_result;
				i -= vlen_result;
	  			__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
			}
		}
	} else {

		__asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 1, 0, 1)));
		int vlen_result;
		__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (n));

		__asm__ volatile ("vmcs vs1,  %0" : : "r" (*alpha));
		vf(&bli_1v_hwacha_vf_init);

		if ( incx == 1 )
		{
			for ( dim_t i = n; i > 0;)
			{
            			MEMTOUCH(y+offset, float, vlen_result);
            			MEMTOUCH(x+offset, float, vlen_result);
				__asm__ volatile ("vmca va0,  %0" : : "r" (x+offset));
				__asm__ volatile ("vmca va1,  %0" : : "r" (y+offset));
				vf(&bli_sscal2v_unit_hwacha_vf_main);
				offset += vlen_result;
				i -= vlen_result;
	  			__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
			}
		}
		else
		{
			__asm__ volatile ("vmca va2,  %0" : : "r" (incx*sizeof(float)));
			__asm__ volatile ("vmca va3,  %0" : : "r" (incy*sizeof(float)));
			for ( dim_t i = n; i > 0;)
			{
            			MEMTOUCH(y+offset*incy, float, vlen_result*incy);
            			MEMTOUCH(x+offset*incx, float, vlen_result*incx);
				__asm__ volatile ("vmca va0,  %0" : : "r" (x+offset*incx));
				__asm__ volatile ("vmca va1,  %0" : : "r" (y+offset*incy));
				vf(&bli_sscal2v_stride_hwacha_vf_main);
				offset += vlen_result;
				i -= vlen_result;
	  			__asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (i));
			}
		}
	}
	__asm__ volatile ("fence" ::: "memory");
}
