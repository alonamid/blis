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
#include "include/gemmini_params.h"

void bli_ssetv_lowprec
     (
       conj_t           conjalpha,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
	if ( bli_zero_dim1( n ) ) return;

	if (bli_cntx_lowprec_in_use(cntx))
	{
		elem_t* restrict x_elem = (elem_t*)x;

		if ( bli_seq0( *alpha ) )
		{
			if ( incx == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					x_elem[i] = 0;
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					*x_elem = 0;
	
					x_elem += incx;
				}
			}
		}
		else
		{
			float alpha_conj;
			elem_t alpha_conj_elem;
	
			bli_scopycjs( conjalpha, *alpha, alpha_conj );
			bli_tolowprec(alpha_conj, alpha_conj_elem);
	
			if ( incx == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					x_elem[i] = alpha_conj_elem;
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					*x_elem = alpha_conj_elem;
	
					x_elem += incx;
				}
			}
		}
	} else {
		if ( bli_seq0( *alpha ) )
		{
			if ( incx == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_sset0s( x[i] );
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_sset0s( *x );
	
					x += incx;
				}
			}
		}
		else
		{
			float alpha_conj;
	
			bli_scopycjs( conjalpha, *alpha, alpha_conj );
	
			if ( incx == 1 )
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_scopys( alpha_conj, x[i] );
				}
			}
			else
			{
				for ( dim_t i = 0; i < n; ++i )
				{
					bli_scopys( alpha_conj, *x );
	
					x += incx;
				}
			}
		}
	}
}
