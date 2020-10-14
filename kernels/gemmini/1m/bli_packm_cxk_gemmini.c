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

#define FP32_SIG_BITS 23
#define FP32_EXP_BITS 8

typedef union {
  float f;
  struct {
    unsigned int mantisa : FP32_SIG_BITS;
    unsigned int exponent : FP32_EXP_BITS;
    unsigned int sign : 1;
  } parts;
} float_cast;

#define packToF16UI( sign, exp, sig ) (((uint16_t) (sign)<<15) + ((uint16_t) (exp)<<(ELEM_T_SIG_BITS - 1)) + (sig))

#ifdef ELEM_T_IS_LOWPREC_FLOAT
#define bli_scopysconvert( a, b ) \
{ \
        float_cast src_bits = { (a) }; \
	(b) = packToF16UI( src_bits.parts.sign, src_bits.parts.exponent, src_bits.parts.mantisa >> (FP32_SIG_BITS - ELEM_T_SIG_BITS) ); \
}
#else
#define bli_scopysconvert( a, b )  bli_scopys(a, b)
#endif


//unrolled 32
void bli_spackm_gemmini_32xk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim,
       dim_t            n,
       dim_t            n_max,
       void*   restrict kappa,
       void*   restrict a, inc_t inca, inc_t lda,
       void*   restrict p,             inc_t ldp,
       cntx_t* restrict cntx
     )
{
    float* restrict kappa_cast = kappa;
    float* restrict alpha1     = a;
    float* restrict pi1        = p;

    dim_t           mnr        = 32;

    if ( cdim == mnr ) //the "standard" case for packing, where a big matrix needs to be packed into panels
    {
      if (*kappa_cast == 0) // no kappa_cast
      {
	if ( bli_is_conj( conja ) )
        {
          for ( dim_t k = n; k != 0; --k )
          {
            bli_scopyjs(*(alpha1 + 0*inca), *(pi1 + 0))
            bli_scopyjs(*(alpha1 + 1*inca), *(pi1 + 1))
            bli_scopyjs(*(alpha1 + 2*inca), *(pi1 + 2))
            bli_scopyjs(*(alpha1 + 3*inca), *(pi1 + 3))
            bli_scopyjs(*(alpha1 + 4*inca), *(pi1 + 4))
            bli_scopyjs(*(alpha1 + 5*inca), *(pi1 + 5))
            bli_scopyjs(*(alpha1 + 6*inca), *(pi1 + 6))
            bli_scopyjs(*(alpha1 + 7*inca), *(pi1 + 7))
            bli_scopyjs(*(alpha1 + 8*inca), *(pi1 + 8))
            bli_scopyjs(*(alpha1 + 9*inca), *(pi1 + 9))
            bli_scopyjs(*(alpha1 + 10*inca), *(pi1 + 10))
            bli_scopyjs(*(alpha1 + 11*inca), *(pi1 + 11))
            bli_scopyjs(*(alpha1 + 12*inca), *(pi1 + 12))
            bli_scopyjs(*(alpha1 + 13*inca), *(pi1 + 13))
            bli_scopyjs(*(alpha1 + 14*inca), *(pi1 + 14))
            bli_scopyjs(*(alpha1 + 15*inca), *(pi1 + 15))
            bli_scopyjs(*(alpha1 + 16*inca), *(pi1 + 16))
            bli_scopyjs(*(alpha1 + 17*inca), *(pi1 + 17))
            bli_scopyjs(*(alpha1 + 18*inca), *(pi1 + 18))
            bli_scopyjs(*(alpha1 + 19*inca), *(pi1 + 19))
            bli_scopyjs(*(alpha1 + 20*inca), *(pi1 + 20))
            bli_scopyjs(*(alpha1 + 21*inca), *(pi1 + 21))
            bli_scopyjs(*(alpha1 + 22*inca), *(pi1 + 22))
            bli_scopyjs(*(alpha1 + 23*inca), *(pi1 + 23))
            bli_scopyjs(*(alpha1 + 24*inca), *(pi1 + 24))
            bli_scopyjs(*(alpha1 + 25*inca), *(pi1 + 25))
            bli_scopyjs(*(alpha1 + 26*inca), *(pi1 + 26))
            bli_scopyjs(*(alpha1 + 27*inca), *(pi1 + 27))
            bli_scopyjs(*(alpha1 + 28*inca), *(pi1 + 28))
            bli_scopyjs(*(alpha1 + 29*inca), *(pi1 + 29))
            bli_scopyjs(*(alpha1 + 30*inca), *(pi1 + 30))
            bli_scopyjs(*(alpha1 + 31*inca), *(pi1 + 31))
    
            alpha1 += lda;
            pi1    += ldp;
          }
        }
        else
        {
          for ( dim_t k = n; k != 0; --k )
          {
            bli_scopysconvert(*(alpha1 + 0*inca), *(pi1 + 0))
            bli_scopysconvert(*(alpha1 + 1*inca), *(pi1 + 1))
            bli_scopysconvert(*(alpha1 + 2*inca), *(pi1 + 2))
            bli_scopysconvert(*(alpha1 + 3*inca), *(pi1 + 3))
            bli_scopysconvert(*(alpha1 + 4*inca), *(pi1 + 4))
            bli_scopysconvert(*(alpha1 + 5*inca), *(pi1 + 5))
            bli_scopysconvert(*(alpha1 + 6*inca), *(pi1 + 6))
            bli_scopysconvert(*(alpha1 + 7*inca), *(pi1 + 7))
            bli_scopysconvert(*(alpha1 + 8*inca), *(pi1 + 8))
            bli_scopysconvert(*(alpha1 + 9*inca), *(pi1 + 9))
            bli_scopysconvert(*(alpha1 + 10*inca), *(pi1 + 10))
            bli_scopysconvert(*(alpha1 + 11*inca), *(pi1 + 11))
            bli_scopysconvert(*(alpha1 + 12*inca), *(pi1 + 12))
            bli_scopysconvert(*(alpha1 + 13*inca), *(pi1 + 13))
            bli_scopysconvert(*(alpha1 + 14*inca), *(pi1 + 14))
            bli_scopysconvert(*(alpha1 + 15*inca), *(pi1 + 15))
            bli_scopysconvert(*(alpha1 + 16*inca), *(pi1 + 16))
            bli_scopysconvert(*(alpha1 + 17*inca), *(pi1 + 17))
            bli_scopysconvert(*(alpha1 + 18*inca), *(pi1 + 18))
            bli_scopysconvert(*(alpha1 + 19*inca), *(pi1 + 19))
            bli_scopysconvert(*(alpha1 + 20*inca), *(pi1 + 20))
            bli_scopysconvert(*(alpha1 + 21*inca), *(pi1 + 21))
            bli_scopysconvert(*(alpha1 + 22*inca), *(pi1 + 22))
            bli_scopysconvert(*(alpha1 + 23*inca), *(pi1 + 23))
            bli_scopysconvert(*(alpha1 + 24*inca), *(pi1 + 24))
            bli_scopysconvert(*(alpha1 + 25*inca), *(pi1 + 25))
            bli_scopysconvert(*(alpha1 + 26*inca), *(pi1 + 26))
            bli_scopysconvert(*(alpha1 + 27*inca), *(pi1 + 27))
            bli_scopysconvert(*(alpha1 + 28*inca), *(pi1 + 28))
            bli_scopysconvert(*(alpha1 + 29*inca), *(pi1 + 29))
            bli_scopysconvert(*(alpha1 + 30*inca), *(pi1 + 30))
            bli_scopysconvert(*(alpha1 + 31*inca), *(pi1 + 31))
/*
            bli_scopys(*(alpha1 + 0*inca), *(pi1 + 0))
            bli_scopys(*(alpha1 + 1*inca), *(pi1 + 1))
            bli_scopys(*(alpha1 + 2*inca), *(pi1 + 2))
            bli_scopys(*(alpha1 + 3*inca), *(pi1 + 3))
            bli_scopys(*(alpha1 + 4*inca), *(pi1 + 4))
            bli_scopys(*(alpha1 + 5*inca), *(pi1 + 5))
            bli_scopys(*(alpha1 + 6*inca), *(pi1 + 6))
            bli_scopys(*(alpha1 + 7*inca), *(pi1 + 7))
            bli_scopys(*(alpha1 + 8*inca), *(pi1 + 8))
            bli_scopys(*(alpha1 + 9*inca), *(pi1 + 9))
            bli_scopys(*(alpha1 + 10*inca), *(pi1 + 10))
            bli_scopys(*(alpha1 + 11*inca), *(pi1 + 11))
            bli_scopys(*(alpha1 + 12*inca), *(pi1 + 12))
            bli_scopys(*(alpha1 + 13*inca), *(pi1 + 13))
            bli_scopys(*(alpha1 + 14*inca), *(pi1 + 14))
            bli_scopys(*(alpha1 + 15*inca), *(pi1 + 15))
            bli_scopys(*(alpha1 + 16*inca), *(pi1 + 16))
            bli_scopys(*(alpha1 + 17*inca), *(pi1 + 17))
            bli_scopys(*(alpha1 + 18*inca), *(pi1 + 18))
            bli_scopys(*(alpha1 + 19*inca), *(pi1 + 19))
            bli_scopys(*(alpha1 + 20*inca), *(pi1 + 20))
            bli_scopys(*(alpha1 + 21*inca), *(pi1 + 21))
            bli_scopys(*(alpha1 + 22*inca), *(pi1 + 22))
            bli_scopys(*(alpha1 + 23*inca), *(pi1 + 23))
            bli_scopys(*(alpha1 + 24*inca), *(pi1 + 24))
            bli_scopys(*(alpha1 + 25*inca), *(pi1 + 25))
            bli_scopys(*(alpha1 + 26*inca), *(pi1 + 26))
            bli_scopys(*(alpha1 + 27*inca), *(pi1 + 27))
            bli_scopys(*(alpha1 + 28*inca), *(pi1 + 28))
            bli_scopys(*(alpha1 + 29*inca), *(pi1 + 29))
            bli_scopys(*(alpha1 + 30*inca), *(pi1 + 30))
            bli_scopys(*(alpha1 + 31*inca), *(pi1 + 31))
*/    
            alpha1 += lda;
            pi1    += ldp;
          }
        }
      }
      else //there is a kappa_cast
      {
        for ( dim_t k = n; k != 0; --k )            
        {
          bli_sscal2s( *kappa_cast, *(alpha1 + 0*inca), *(pi1 + 0) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 1*inca), *(pi1 + 1) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 2*inca), *(pi1 + 2) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 3*inca), *(pi1 + 3) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 4*inca), *(pi1 + 4) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 5*inca), *(pi1 + 5) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 6*inca), *(pi1 + 6) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 7*inca), *(pi1 + 7) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 8*inca), *(pi1 + 8) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 9*inca), *(pi1 + 9) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 10*inca), *(pi1 + 10) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 11*inca), *(pi1 + 11) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 12*inca), *(pi1 + 12) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 13*inca), *(pi1 + 13) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 14*inca), *(pi1 + 14) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 15*inca), *(pi1 + 15) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 16*inca), *(pi1 + 16) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 17*inca), *(pi1 + 17) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 18*inca), *(pi1 + 18) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 19*inca), *(pi1 + 19) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 20*inca), *(pi1 + 20) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 21*inca), *(pi1 + 21) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 22*inca), *(pi1 + 22) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 23*inca), *(pi1 + 23) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 24*inca), *(pi1 + 24) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 25*inca), *(pi1 + 25) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 26*inca), *(pi1 + 26) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 27*inca), *(pi1 + 27) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 28*inca), *(pi1 + 28) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 29*inca), *(pi1 + 29) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 30*inca), *(pi1 + 30) );
          bli_sscal2s( *kappa_cast, *(alpha1 + 31*inca), *(pi1 + 31) );

          alpha1 += lda;
          pi1    += ldp;
        }
      }
    }
    else // cdim < mnr
    // if the matrix size is smaller than the panel size
    // then copy as-is and scale by kappa (not need to pack)
    {

       bli_sscal2m_ex(
               0,
               BLIS_NONUNIT_DIAG,
               BLIS_DENSE,
               ( trans_t )conja,
               cdim,
               n,
               kappa,
               a, inca, lda,
               p,    1, ldp,
               cntx,
               NULL
             );

      // pad with zeros if the panel size is greater than the matrix size

      const dim_t     i      = cdim;
      const dim_t     m_edge = mnr - cdim;
      const dim_t     n_edge = n_max;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (i  )*1;

      bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);

    }

    // pad with zeros if there is a difference between the logical size and physical size
    if ( n < n_max )
    {
      const dim_t     j      = n;
      const dim_t     m_edge = mnr;
      const dim_t     n_edge = n_max - n;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (j  )*ldp;

      bli_sset0s_mxn(m_edge, n_edge, p_edge, 1, ldp);
    }

}


//generic, but not unrolled
//should be functional, but lower-perf
/*
void bli_spackm_gemmini_cxk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim,
       dim_t            n,
       dim_t            n_max,
       void*   restrict kappa,
       void*   restrict a, inc_t inca, inc_t lda,
       void*   restrict p,             inc_t ldp,
       cntx_t* restrict cntx
     )
{
    float* restrict kappa_cast = kappa;
    float* restrict alpha1     = a;
    float* restrict pi1        = p;

    dim_t           mnr        = BLIS_MR;

    if ( cdim == mnr ) //the "standard" case for packing, where a big matrix needs to be packed into panels
    {
      if (*kappa_cast == 0) // no kappa_cast
      {
	if ( bli_is_conj( conja ) )
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < mnr; i++)
            {
              bli_scopyjs(*(alpha1 + i*inca), *(pi1 + i))
            } 
            alpha1 += lda;
            pi1    += ldp;
          }
        }
        else
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < mnr; i++)
            {
              bli_scopysconvert(*(alpha1 + i*inca), *(pi1 + i))
              //bli_scopys(*(alpha1 + i*inca), *(pi1 + i))
            }    
            alpha1 += lda;
            pi1    += ldp;
          }
        }
      }
      else //there is a kappa_cast
      {
        for ( dim_t k = n; k != 0; --k )            
        {
          for ( int i = 0; i < mnr; i++)
          {
            bli_sscal2s( *kappa_cast, *(alpha1 + i*inca), *(pi1 + i) );
          }

          alpha1 += lda;
          pi1    += ldp;
        }
      }
    }
    else // cdim < mnr
    // if the matrix size is smaller than the panel size
    // then copy as-is and scale by kappa (not need to pack)
    {

       bli_sscal2m_ex(
               0,
               BLIS_NONUNIT_DIAG,
               BLIS_DENSE,
               ( trans_t )conja,
               cdim,
               n,
               kappa,
               a, inca, lda,
               p,    1, ldp,
               cntx,
               NULL
             );

      // pad with zeros if the panel size is greater than the matrix size

      const dim_t     i      = cdim;
      const dim_t     m_edge = mnr - cdim;
      const dim_t     n_edge = n_max;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (i  )*1;

      bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);

    }

    // pad with zeros if there is a difference between the logical size and physical size
    if ( n < n_max )
    {
      const dim_t     j      = n;
      const dim_t     m_edge = mnr;
      const dim_t     n_edge = n_max - n;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (j  )*ldp;

      bli_sset0s_mxn(m_edge, n_edge, p_edge, 1, ldp);
    }

}
*/
