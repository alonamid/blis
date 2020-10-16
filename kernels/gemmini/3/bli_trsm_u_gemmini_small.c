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

#define FP32_SIG_BITS 23
#define FP32_EXP_BITS 8

typedef union {
  uint16_t f;
  struct {
    unsigned int mantisa : ELEM_T_SIG_BITS;
    unsigned int exponent : ELEM_T_EXP_BITS;
    unsigned int sign : 1;
  } parts;
} lowprec_cast;

typedef union {
  float f;
  struct {
    unsigned int mantisa : FP32_SIG_BITS;
    unsigned int exponent : FP32_EXP_BITS;
    unsigned int sign : 1;
  } parts;
} float_cast;

#define packToF32UI( sign, exp, sig ) (((uint32_t) (sign)<<31) + ((uint32_t) (exp)<<(FP32_SIG_BITS)) + (sig))
#define packToF16UI( sign, exp, sig ) (((uint16_t) (sign)<<15) + ((uint16_t) (exp)<<(ELEM_T_SIG_BITS - 1)) + (sig))

#ifdef ELEM_T_IS_LOWPREC_FLOAT
#define bli_tofloat( a, b ) \
{ \
        lowprec_cast src_bits = { (a) }; \
        (b) = packToF32UI( src_bits.parts.sign, src_bits.parts.exponent, src_bits.parts.mantisa << (FP32_SIG_BITS - ELEM_T_SIG_BITS) ); \
}
#else
#define bli_tofloat( a, b)  bli_scopys(a, b)
#endif

#ifdef ELEM_T_IS_LOWPREC_FLOAT
#define bli_tolowprec( a, b ) \
{ \
        float_cast src_bits = { (a) }; \
        (b) = packToF16UI( src_bits.parts.sign, src_bits.parts.exponent, src_bits.parts.mantisa >> (FP32_SIG_BITS - ELEM_T_SIG_BITS) ); \
}
#else
#define bli_tolowprec( a, b )  bli_scopys(a, b)
#endif

void bli_strsm_u_gemmini_small
     (
       float*  restrict a11,
       float*  restrict b11,
       float*  restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
/*
  Template trsm_u micro-kernel implementation

  This function contains a template implementation for a double-precision
  complex trsm micro-kernel, coded in C, which can serve as the starting point
  for one to write an optimized micro-kernel on an arbitrary architecture.
  (We show a template implementation for only double-precision complex because
  the templates for the other three floating-point types would be nearly
  identical.)

  This micro-kernel performs the following operation:

    C11 := inv(A11) * B11

  where A11 is MR x MR and upper triangular, B11 is MR x NR, and C11 is
  MR x NR.

  For more info, please refer to the BLIS website's wiki on kernels:

    https://github.com/flame/blis/wiki/KernelsHowTo

  and/or contact the blis-devel mailing list.

  -FGVZ
*/
        const num_t        dt     = BLIS_FLOAT;

	const dim_t        mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t        nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );

	const inc_t        packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx );
	const inc_t        packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );

	const dim_t        m      = mr;
	const dim_t        n      = nr;

	const inc_t        rs_a   = 1;
	const inc_t        cs_a   = packmr;

	const inc_t        rs_b   = packnr;
	const inc_t        cs_b   = 1;

	dim_t              iter, i, j, l;
	dim_t              n_behind;

	elem_t* restrict alpha11;
	float            alpha11_f;
	elem_t* restrict a12t;
	elem_t* restrict alpha12;
	float            alpha12_f;
	elem_t* restrict X2;
	elem_t* restrict x1;
	elem_t* restrict x21;
	elem_t* restrict chi21;
	elem_t* restrict chi11;
	float            chi21_f;
	float            chi11_f;
	float* restrict  gamma11;
	float            rho11;

	for ( iter = 0; iter < m; ++iter )
	{
		i        = m - iter - 1;
		n_behind = iter;
		alpha11  = (elem_t*)a11 + (i  )*rs_a + (i  )*cs_a;
		a12t     = (elem_t*)a11 + (i  )*rs_a + (i+1)*cs_a;
		x1       = (elem_t*)b11 + (i  )*rs_b + (0  )*cs_b;
		X2       = (elem_t*)b11 + (i+1)*rs_b + (0  )*cs_b;

		/* x1 = x1 - a12t * X2; */
		/* x1 = x1 / alpha11; */
		for ( j = 0; j < n; ++j )
		{
			chi11   = x1  + (0  )*rs_b + (j  )*cs_b;
			x21     = X2  + (0  )*rs_b + (j  )*cs_b;
			gamma11 = c11 + (i  )*rs_c + (j  )*cs_c;

			/* chi11 = chi11 - a12t * x21; */
			bli_sset0s( rho11 );
			for ( l = 0; l < n_behind; ++l )
			{
				alpha12 = a12t + (l  )*cs_a;
				chi21   = x21  + (l  )*rs_b;

				bli_tofloat(*alpha12, alpha12_f);
				bli_tofloat(*chi21, chi21_f);
				bli_saxpys( alpha12_f, chi21_f, rho11 );
			}
			bli_tofloat(*chi11, chi11_f);
			bli_ssubs( rho11, chi11_f );

			/* chi11 = chi11 / alpha11; */
			/* NOTE: The INVERSE of alpha11 (1.0/alpha11) is stored instead
			   of alpha11, so we can multiply rather than divide. We store
			   the inverse of alpha11 intentionally to avoid expensive
			   division instructions within the micro-kernel. */
			bli_tofloat(*alpha11, alpha11_f);
			bli_sscals( alpha11_f, chi11_f );

			/* Output final result to matrix C. */
			bli_scopys( chi11_f, *gamma11 );
                        bli_tolowprec(chi11_f, *chi11);
		}
	}
}

