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

#ifdef ELEM_T_IS_LOWPREC_FLOAT
#define bli_scopysconvert( a, b ) \
{ \
    float_cast tmp = { (a) }; \
    (b) = (elem_t)(tmp.bits >> (FP32_SIG_BITS - (ELEM_T_SIG_BITS - 1))); \
}
#else
#define bli_scopysconvert( a, b )  bli_scopys(a, b)
#endif

void bli_scopyv_lowprec
     (
       conj_t           conjx,
       dim_t            n,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	if ( bli_zero_dim1( n ) ) return;

	if (bli_cntx_lowprec_in_use(cntx) && bli_cntx_lowprec_elem_out(cntx))
	{
		elem_t* restrict y_elem = (elem_t*)y;

		if ( bli_is_conj( conjx ) )
		{
			if ( incx == 1 && incy == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_scopysconvert( x[i], y_elem[i] );
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_scopysconvert( *x, *y_elem );

					x += incx;
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
					bli_scopysconvert( x[i], y_elem[i] );
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_scopysconvert( *x, *y_elem );

					x += incx;
					y_elem += incy;
				}
			}
		}

	}
	else
	{
		if ( bli_is_conj( conjx ) )
		{
			if ( incx == 1 && incy == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_scopyjs( x[i], y[i] );
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_scopyjs( *x, *y );

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
					bli_scopys( x[i], y[i] );
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_scopys( *x, *y );

					x += incx;
					y += incy;
				}
			}
		}
	}
}
