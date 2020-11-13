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

void bli_sscal2v_lowprec
     (
       conj_t           conjx,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	if ( bli_zero_dim1( n ) ) return;

	if ( bli_seq0( *alpha ) )
	{
		/* If alpha is zero, use setv. */
		float* zero = bli_s0;
	
		bli_ssetv_lowprec
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  zero,
		  y, incy,
		  cntx
		);
		return;
	}

	else if ( bli_seq1( *alpha ) )
	{
		/* If alpha is one, use copyv. */

		if (bli_cntx_lowprec_in_use(cntx))
		{
			for (int i=0; i < n; i++) 
				*((elem_t*)y+ i*incy) = *((elem_t*)x + i*incx);
		}
		else
		{
			for (int i=0; i < n; i++) 
				*(y+ i*incy) = *(x + i*incx);

		}

		return;
	}

	if (bli_cntx_lowprec_in_use(cntx))
	{
		elem_t* restrict x_elem = (elem_t*)x;
		elem_t* restrict y_elem = (elem_t*)y;
		float x_f;
		float y_f;

		if ( bli_is_conj( conjx ) )
		{
			if ( incx == 1 && incy == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_tofloat(x_elem[i], x_f);
					bli_sscal2js( *alpha, x_f, y_f );
					bli_tolowprec(y_f, y_elem[i]);
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_tofloat(*x_elem, x_f);
					bli_sscal2js( *alpha, x_f, y_f );
					bli_tolowprec(y_f, *y_elem);
	
					x_elem += incx;
					y_elem += incy;
				}
			}
		}
		else
		{
			if ( incx == 1 && incy == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_tofloat(x_elem[i], x_f);
					bli_sscal2s( *alpha, x_f, y_f );
					bli_tolowprec(y_f, y_elem[i]);
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_tofloat(*x_elem, x_f);
					bli_sscal2s( *alpha, x_f, y_f );
					bli_tolowprec(y_f, *y_elem);
	
					x_elem += incx;
					y_elem += incy;
				}
			}
		}
	} else {
		if ( bli_is_conj( conjx ) )
		{
			if ( incx == 1 && incy == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_sscal2js( *alpha, x[i], y[i] );
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_sscal2js( *alpha, *x, *y );
	
					x += incx;
					y += incy;
				}
			}
		}
		else
		{
			if ( incx == 1 && incy == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_sscal2s( *alpha, x[i], y[i] );
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_sscal2s( *alpha, *x, *y );
	
					x += incx;
					y += incy;
				}
			}
		}
	}
}
