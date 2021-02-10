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

#define bli_scopysconvert( a, b ) bli_tolowprec( (a), (b) )
#define bli_sscal2sconvert(x, a, b ) bli_tolowprec( (x) * (a), (b) )

//unrolled 32
void bli_spackm_gemmini_32xk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim,
       dim_t            n,
       dim_t            n_max,
       float*   restrict kappa,
       float*   restrict a, inc_t inca, inc_t lda,
       float*   restrict p,             inc_t ldp,
       cntx_t* restrict cntx
     )
{
    float*  restrict kappa_cast = kappa;
    float*  restrict alpha1     = a;
    float* restrict pi1       = p;
    elem_t* restrict pi1_lp       = (elem_t*)p;

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
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	  if (bli_cntx_lowprec_in_use(cntx))
	  {
            for ( dim_t k = n; k != 0; --k )
            {
              bli_scopysconvert(*(alpha1 + 0*inca), *(pi1_lp + 0))
              bli_scopysconvert(*(alpha1 + 1*inca), *(pi1_lp + 1))
              bli_scopysconvert(*(alpha1 + 2*inca), *(pi1_lp + 2))
              bli_scopysconvert(*(alpha1 + 3*inca), *(pi1_lp + 3))
              bli_scopysconvert(*(alpha1 + 4*inca), *(pi1_lp + 4))
              bli_scopysconvert(*(alpha1 + 5*inca), *(pi1_lp + 5))
              bli_scopysconvert(*(alpha1 + 6*inca), *(pi1_lp + 6))
              bli_scopysconvert(*(alpha1 + 7*inca), *(pi1_lp + 7))
              bli_scopysconvert(*(alpha1 + 8*inca), *(pi1_lp + 8))
              bli_scopysconvert(*(alpha1 + 9*inca), *(pi1_lp + 9))
              bli_scopysconvert(*(alpha1 + 10*inca), *(pi1_lp + 10))
              bli_scopysconvert(*(alpha1 + 11*inca), *(pi1_lp + 11))
              bli_scopysconvert(*(alpha1 + 12*inca), *(pi1_lp + 12))
              bli_scopysconvert(*(alpha1 + 13*inca), *(pi1_lp + 13))
              bli_scopysconvert(*(alpha1 + 14*inca), *(pi1_lp + 14))
              bli_scopysconvert(*(alpha1 + 15*inca), *(pi1_lp + 15))
              bli_scopysconvert(*(alpha1 + 16*inca), *(pi1_lp + 16))
              bli_scopysconvert(*(alpha1 + 17*inca), *(pi1_lp + 17))
              bli_scopysconvert(*(alpha1 + 18*inca), *(pi1_lp + 18))
              bli_scopysconvert(*(alpha1 + 19*inca), *(pi1_lp + 19))
              bli_scopysconvert(*(alpha1 + 20*inca), *(pi1_lp + 20))
              bli_scopysconvert(*(alpha1 + 21*inca), *(pi1_lp + 21))
              bli_scopysconvert(*(alpha1 + 22*inca), *(pi1_lp + 22))
              bli_scopysconvert(*(alpha1 + 23*inca), *(pi1_lp + 23))
              bli_scopysconvert(*(alpha1 + 24*inca), *(pi1_lp + 24))
              bli_scopysconvert(*(alpha1 + 25*inca), *(pi1_lp + 25))
              bli_scopysconvert(*(alpha1 + 26*inca), *(pi1_lp + 26))
              bli_scopysconvert(*(alpha1 + 27*inca), *(pi1_lp + 27))
              bli_scopysconvert(*(alpha1 + 28*inca), *(pi1_lp + 28))
              bli_scopysconvert(*(alpha1 + 29*inca), *(pi1_lp + 29))
              bli_scopysconvert(*(alpha1 + 30*inca), *(pi1_lp + 30))
              bli_scopysconvert(*(alpha1 + 31*inca), *(pi1_lp + 31))

              alpha1 += lda;
              pi1_lp += ldp;
	    }
	  }
          else
#endif
	  {
            for ( dim_t k = n; k != 0; --k )
            {
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

              alpha1 += lda;
              pi1    += ldp;
	    }
          }
        }
      }
      else //there is a kappa_cast
      {
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
	{
          for ( dim_t k = n; k != 0; --k )
          {
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 0*inca), *(pi1_lp + 0) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 1*inca), *(pi1_lp + 1) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 2*inca), *(pi1_lp + 2) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 3*inca), *(pi1_lp + 3) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 4*inca), *(pi1_lp + 4) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 5*inca), *(pi1_lp + 5) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 6*inca), *(pi1_lp + 6) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 7*inca), *(pi1_lp + 7) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 8*inca), *(pi1_lp + 8) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 9*inca), *(pi1_lp + 9) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 10*inca), *(pi1_lp + 10) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 11*inca), *(pi1_lp + 11) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 12*inca), *(pi1_lp + 12) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 13*inca), *(pi1_lp + 13) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 14*inca), *(pi1_lp + 14) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 15*inca), *(pi1_lp + 15) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 16*inca), *(pi1_lp + 16) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 17*inca), *(pi1_lp + 17) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 18*inca), *(pi1_lp + 18) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 19*inca), *(pi1_lp + 19) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 20*inca), *(pi1_lp + 20) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 21*inca), *(pi1_lp + 21) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 22*inca), *(pi1_lp + 22) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 23*inca), *(pi1_lp + 23) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 24*inca), *(pi1_lp + 24) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 25*inca), *(pi1_lp + 25) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 26*inca), *(pi1_lp + 26) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 27*inca), *(pi1_lp + 27) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 28*inca), *(pi1_lp + 28) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 29*inca), *(pi1_lp + 29) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 30*inca), *(pi1_lp + 30) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 31*inca), *(pi1_lp + 31) );

            alpha1 += lda;
            pi1_lp    += ldp;
          }
        }
        else
#endif
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
    }
    else // cdim < mnr
    // if the matrix size is smaller than the panel size
    // then copy as-is and scale by kappa (not need to pack)
    {

#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < cdim; i++)
            {
              bli_sscal2sconvert( *kappa_cast, *(alpha1 + i*inca), *(pi1_lp + i) );
            }

            alpha1 += lda;
            pi1_lp += ldp;
          }
        }
        else
#endif
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < cdim; i++)
            {
              bli_sscal2s( *kappa_cast, *(alpha1 + i*inca), *(pi1 + i) );
            }

            alpha1 += lda;
            pi1    += ldp;
          }
        }

      // pad with zeros if the panel size is greater than the matrix size

      const dim_t     i      = cdim;
      const dim_t     m_edge = mnr - cdim;
      const dim_t     n_edge = n_max;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (i  )*1;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (i  )*1;
        for ( dim_t jj = 0; jj < n_edge; ++jj ) {
        for ( dim_t ii = 0; ii < m_edge; ++ii ) {
          *(p_edge_lp + ii + jj*ldp) = 0;
        }}
      }
      else
#endif
      {
        bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
      }
    }

    // pad with zeros if there is a difference between the logical size and physical size
    if ( n < n_max )
    {
      const dim_t     j      = n;
      const dim_t     m_edge = mnr;
      const dim_t     n_edge = n_max - n;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (j  )*ldp;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (j  )*ldp;
        for ( dim_t jj = 0; jj < n_edge; ++jj )
        for ( dim_t ii = 0; ii < m_edge; ++ii )
          *(p_edge_lp + ii + jj*ldp) = 0;
      }
      else
#endif
      {
        bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
      }
    }

}

//unrolled 4
void bli_spackm_gemmini_4xk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim,
       dim_t            n,
       dim_t            n_max,
       float*   restrict kappa,
       float*   restrict a, inc_t inca, inc_t lda,
       float*   restrict p,             inc_t ldp,
       cntx_t* restrict cntx
     )
{
    float*  restrict kappa_cast = kappa;
    float*  restrict alpha1     = a;
    float* restrict pi1         = p;
    elem_t* restrict pi1_lp       = (elem_t*)p;

    dim_t           mnr        = 4;

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

            alpha1 += lda;
            pi1    += ldp;
          }
        }
        else
        {
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	  if (bli_cntx_lowprec_in_use(cntx))
	  {
            for ( dim_t k = n; k != 0; --k )
            {
              bli_scopysconvert(*(alpha1 + 0*inca), *(pi1_lp + 0))
              bli_scopysconvert(*(alpha1 + 1*inca), *(pi1_lp + 1))
              bli_scopysconvert(*(alpha1 + 2*inca), *(pi1_lp + 2))
              bli_scopysconvert(*(alpha1 + 3*inca), *(pi1_lp + 3))

              alpha1 += lda;
              pi1_lp += ldp;
            }
          }
          else
#endif
          {
            for ( dim_t k = n; k != 0; --k )
            {
              bli_scopys(*(alpha1 + 0*inca), *(pi1 + 0))
              bli_scopys(*(alpha1 + 1*inca), *(pi1 + 1))
              bli_scopys(*(alpha1 + 2*inca), *(pi1 + 2))
              bli_scopys(*(alpha1 + 3*inca), *(pi1 + 3))

              alpha1 += lda;
              pi1    += ldp;
            }
          }
        }
      }
      else //there is a kappa_cast
      {
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
	{
          for ( dim_t k = n; k != 0; --k )
          {
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 0*inca), *(pi1_lp + 0) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 1*inca), *(pi1_lp + 1) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 2*inca), *(pi1_lp + 2) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 3*inca), *(pi1_lp + 3) );

            alpha1 += lda;
            pi1_lp += ldp;
          }
        }
        else
#endif
        {
          for ( dim_t k = n; k != 0; --k )
          {
            bli_sscal2s( *kappa_cast, *(alpha1 + 0*inca), *(pi1 + 0) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 1*inca), *(pi1 + 1) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 2*inca), *(pi1 + 2) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 3*inca), *(pi1 + 3) );

            alpha1 += lda;
            pi1    += ldp;
          }
        }
      }
    }
    else // cdim < mnr
    // if the matrix size is smaller than the panel size
    // then copy as-is and scale by kappa (not need to pack)
    {

#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < cdim; i++)
            {
              bli_sscal2sconvert( *kappa_cast, *(alpha1 + i*inca), *(pi1_lp + i) );
            }

            alpha1 += lda;
            pi1_lp += ldp;
          }
        }
        else
#endif
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < cdim; i++)
            {
              bli_sscal2s( *kappa_cast, *(alpha1 + i*inca), *(pi1 + i) );
            }

            alpha1 += lda;
            pi1    += ldp;
          }
        }

      // pad with zeros if the panel size is greater than the matrix size

      const dim_t     i      = cdim;
      const dim_t     m_edge = mnr - cdim;
      const dim_t     n_edge = n_max;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (i  )*1;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (i  )*1;
        for ( dim_t jj = 0; jj < n_edge; ++jj ) {
        for ( dim_t ii = 0; ii < m_edge; ++ii ) {
          *(p_edge_lp + ii + jj*ldp) = 0;
        }}
      }
      else
#endif
      {
        bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
      }
    }

    // pad with zeros if there is a difference between the logical size and physical size
    if ( n < n_max )
    {
      const dim_t     j      = n;
      const dim_t     m_edge = mnr;
      const dim_t     n_edge = n_max - n;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (j  )*ldp;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (j  )*ldp;
        for ( dim_t jj = 0; jj < n_edge; ++jj )
        for ( dim_t ii = 0; ii < m_edge; ++ii )
          *(p_edge_lp + ii + jj*ldp) = 0;
      }
      else
#endif
      {
        bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
      }
    }

}


//generic, but not unrolled
//should be functional, but lower-perf
void bli_spackm_gemmini_cxk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim,
       dim_t            n,
       dim_t            n_max,
       float*   restrict kappa,
       float*   restrict a, inc_t inca, inc_t lda,
       float*   restrict p,             inc_t ldp,
       cntx_t* restrict cntx
     )
{
    float*  restrict kappa_cast = kappa;
    float*  restrict alpha1     = a;
    float*  restrict pi1        = p;
    elem_t* restrict pi1_lp     = (elem_t*)p;

    dim_t           mnr        = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_MR, cntx );;

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
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	  if (bli_cntx_lowprec_in_use(cntx))
	  {
            for ( dim_t k = n; k != 0; --k )
            {
              for ( int i = 0; i < mnr; i++)
              {
                bli_scopysconvert(*(alpha1 + i*inca), *(pi1_lp + i))
              }
              alpha1 += lda;
              pi1_lp += ldp;
            }
          }
          else
#endif
          {
            for ( dim_t k = n; k != 0; --k )
            {
              for ( int i = 0; i < mnr; i++)
              {
                bli_scopys(*(alpha1 + i*inca), *(pi1 + i))
              }
              alpha1 += lda;
              pi1    += ldp;
            }
          }
        }
      }
      else //there is a kappa_cast
      {
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
	{
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < mnr; i++)
            {
              bli_sscal2sconvert( *kappa_cast, *(alpha1 + i*inca), *(pi1_lp + i) );
            }

            alpha1 += lda;
            pi1_lp += ldp;
          }
        }
        else
#endif
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
    }
    else // cdim < mnr
    // if the matrix size is smaller than the panel size
    // then copy as-is and scale by kappa (not need to pack)
    {

#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < cdim; i++)
            {
              bli_sscal2sconvert( *kappa_cast, *(alpha1 + i*inca), *(pi1_lp + i) );
            }

            alpha1 += lda;
            pi1_lp += ldp;
          }
        }
        else
#endif
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < cdim; i++)
            {
              bli_sscal2s( *kappa_cast, *(alpha1 + i*inca), *(pi1 + i) );
            }

            alpha1 += lda;
            pi1    += ldp;
          }
        }

      // pad with zeros if the panel size is greater than the matrix size

      const dim_t     i      = cdim;
      const dim_t     m_edge = mnr - cdim;
      const dim_t     n_edge = n_max;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (i  )*1;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (i  )*1;
        for ( dim_t jj = 0; jj < n_edge; ++jj ) {
        for ( dim_t ii = 0; ii < m_edge; ++ii ) {
          *(p_edge_lp + ii + jj*ldp) = 0;
        }}
      }
      else
#endif
      {
        bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
      }
    }

    // pad with zeros if there is a difference between the logical size and physical size
    if ( n < n_max )
    {
      const dim_t     j      = n;
      const dim_t     m_edge = mnr;
      const dim_t     n_edge = n_max - n;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (j  )*ldp;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (j  )*ldp;
        for ( dim_t jj = 0; jj < n_edge; ++jj )
        for ( dim_t ii = 0; ii < m_edge; ++ii )
          *(p_edge_lp + ii + jj*ldp) = 0;
      }
      else
#endif
      {
        bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
      }
    }

}




//unrolled 88
void bli_spackm_gemmini_88xk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim,
       dim_t            n,
       dim_t            n_max,
       float*   restrict kappa,
       float*   restrict a, inc_t inca, inc_t lda,
       float*   restrict p,             inc_t ldp,
       cntx_t* restrict cntx
     )
{
    float*  restrict kappa_cast = kappa;
    float*  restrict alpha1     = a;
    float* restrict pi1       = p;
    elem_t* restrict pi1_lp       = (elem_t*)p;

    dim_t           mnr        = 88;

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
            bli_scopyjs(*(alpha1 + 32*inca), *(pi1 + 32))
            bli_scopyjs(*(alpha1 + 33*inca), *(pi1 + 33))
            bli_scopyjs(*(alpha1 + 34*inca), *(pi1 + 34))
            bli_scopyjs(*(alpha1 + 35*inca), *(pi1 + 35))
            bli_scopyjs(*(alpha1 + 36*inca), *(pi1 + 36))
            bli_scopyjs(*(alpha1 + 37*inca), *(pi1 + 37))
            bli_scopyjs(*(alpha1 + 38*inca), *(pi1 + 38))
            bli_scopyjs(*(alpha1 + 39*inca), *(pi1 + 39))
            bli_scopyjs(*(alpha1 + 40*inca), *(pi1 + 40))
            bli_scopyjs(*(alpha1 + 41*inca), *(pi1 + 41))
            bli_scopyjs(*(alpha1 + 42*inca), *(pi1 + 42))
            bli_scopyjs(*(alpha1 + 43*inca), *(pi1 + 43))
            bli_scopyjs(*(alpha1 + 44*inca), *(pi1 + 44))
            bli_scopyjs(*(alpha1 + 45*inca), *(pi1 + 45))
            bli_scopyjs(*(alpha1 + 46*inca), *(pi1 + 46))
            bli_scopyjs(*(alpha1 + 47*inca), *(pi1 + 47))
            bli_scopyjs(*(alpha1 + 48*inca), *(pi1 + 48))
            bli_scopyjs(*(alpha1 + 49*inca), *(pi1 + 49))
            bli_scopyjs(*(alpha1 + 50*inca), *(pi1 + 50))
            bli_scopyjs(*(alpha1 + 51*inca), *(pi1 + 51))
            bli_scopyjs(*(alpha1 + 52*inca), *(pi1 + 52))
            bli_scopyjs(*(alpha1 + 53*inca), *(pi1 + 53))
            bli_scopyjs(*(alpha1 + 54*inca), *(pi1 + 54))
            bli_scopyjs(*(alpha1 + 55*inca), *(pi1 + 55))
            bli_scopyjs(*(alpha1 + 56*inca), *(pi1 + 56))
            bli_scopyjs(*(alpha1 + 57*inca), *(pi1 + 57))
            bli_scopyjs(*(alpha1 + 58*inca), *(pi1 + 58))
            bli_scopyjs(*(alpha1 + 59*inca), *(pi1 + 59))
            bli_scopyjs(*(alpha1 + 60*inca), *(pi1 + 60))
            bli_scopyjs(*(alpha1 + 61*inca), *(pi1 + 61))
            bli_scopyjs(*(alpha1 + 62*inca), *(pi1 + 62))
            bli_scopyjs(*(alpha1 + 63*inca), *(pi1 + 63))
            bli_scopyjs(*(alpha1 + 64*inca), *(pi1 + 64))
            bli_scopyjs(*(alpha1 + 65*inca), *(pi1 + 65))
            bli_scopyjs(*(alpha1 + 66*inca), *(pi1 + 66))
            bli_scopyjs(*(alpha1 + 67*inca), *(pi1 + 67))
            bli_scopyjs(*(alpha1 + 68*inca), *(pi1 + 68))
            bli_scopyjs(*(alpha1 + 69*inca), *(pi1 + 69))
            bli_scopyjs(*(alpha1 + 70*inca), *(pi1 + 70))
            bli_scopyjs(*(alpha1 + 71*inca), *(pi1 + 71))
            bli_scopyjs(*(alpha1 + 72*inca), *(pi1 + 72))
            bli_scopyjs(*(alpha1 + 73*inca), *(pi1 + 73))
            bli_scopyjs(*(alpha1 + 74*inca), *(pi1 + 74))
            bli_scopyjs(*(alpha1 + 75*inca), *(pi1 + 75))
            bli_scopyjs(*(alpha1 + 76*inca), *(pi1 + 76))
            bli_scopyjs(*(alpha1 + 77*inca), *(pi1 + 77))
            bli_scopyjs(*(alpha1 + 78*inca), *(pi1 + 78))
            bli_scopyjs(*(alpha1 + 79*inca), *(pi1 + 79))
            bli_scopyjs(*(alpha1 + 80*inca), *(pi1 + 80))
            bli_scopyjs(*(alpha1 + 81*inca), *(pi1 + 81))
            bli_scopyjs(*(alpha1 + 82*inca), *(pi1 + 82))
            bli_scopyjs(*(alpha1 + 83*inca), *(pi1 + 83))
            bli_scopyjs(*(alpha1 + 84*inca), *(pi1 + 84))
            bli_scopyjs(*(alpha1 + 85*inca), *(pi1 + 85))
            bli_scopyjs(*(alpha1 + 86*inca), *(pi1 + 86))
            bli_scopyjs(*(alpha1 + 87*inca), *(pi1 + 87))
    
            alpha1 += lda;
            pi1    += ldp;
          }
        }
        else
        {
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	  if (bli_cntx_lowprec_in_use(cntx))
	  {
            for ( dim_t k = n; k != 0; --k )
            {
              bli_scopysconvert(*(alpha1 + 0*inca), *(pi1_lp + 0))
              bli_scopysconvert(*(alpha1 + 1*inca), *(pi1_lp + 1))
              bli_scopysconvert(*(alpha1 + 2*inca), *(pi1_lp + 2))
              bli_scopysconvert(*(alpha1 + 3*inca), *(pi1_lp + 3))
              bli_scopysconvert(*(alpha1 + 4*inca), *(pi1_lp + 4))
              bli_scopysconvert(*(alpha1 + 5*inca), *(pi1_lp + 5))
              bli_scopysconvert(*(alpha1 + 6*inca), *(pi1_lp + 6))
              bli_scopysconvert(*(alpha1 + 7*inca), *(pi1_lp + 7))
              bli_scopysconvert(*(alpha1 + 8*inca), *(pi1_lp + 8))
              bli_scopysconvert(*(alpha1 + 9*inca), *(pi1_lp + 9))
              bli_scopysconvert(*(alpha1 + 10*inca), *(pi1_lp + 10))
              bli_scopysconvert(*(alpha1 + 11*inca), *(pi1_lp + 11))
              bli_scopysconvert(*(alpha1 + 12*inca), *(pi1_lp + 12))
              bli_scopysconvert(*(alpha1 + 13*inca), *(pi1_lp + 13))
              bli_scopysconvert(*(alpha1 + 14*inca), *(pi1_lp + 14))
              bli_scopysconvert(*(alpha1 + 15*inca), *(pi1_lp + 15))
              bli_scopysconvert(*(alpha1 + 16*inca), *(pi1_lp + 16))
              bli_scopysconvert(*(alpha1 + 17*inca), *(pi1_lp + 17))
              bli_scopysconvert(*(alpha1 + 18*inca), *(pi1_lp + 18))
              bli_scopysconvert(*(alpha1 + 19*inca), *(pi1_lp + 19))
              bli_scopysconvert(*(alpha1 + 20*inca), *(pi1_lp + 20))
              bli_scopysconvert(*(alpha1 + 21*inca), *(pi1_lp + 21))
              bli_scopysconvert(*(alpha1 + 22*inca), *(pi1_lp + 22))
              bli_scopysconvert(*(alpha1 + 23*inca), *(pi1_lp + 23))
              bli_scopysconvert(*(alpha1 + 24*inca), *(pi1_lp + 24))
              bli_scopysconvert(*(alpha1 + 25*inca), *(pi1_lp + 25))
              bli_scopysconvert(*(alpha1 + 26*inca), *(pi1_lp + 26))
              bli_scopysconvert(*(alpha1 + 27*inca), *(pi1_lp + 27))
              bli_scopysconvert(*(alpha1 + 28*inca), *(pi1_lp + 28))
              bli_scopysconvert(*(alpha1 + 29*inca), *(pi1_lp + 29))
              bli_scopysconvert(*(alpha1 + 30*inca), *(pi1_lp + 30))
              bli_scopysconvert(*(alpha1 + 31*inca), *(pi1_lp + 31))
              bli_scopysconvert(*(alpha1 + 32*inca), *(pi1_lp + 32))
              bli_scopysconvert(*(alpha1 + 33*inca), *(pi1_lp + 33))
              bli_scopysconvert(*(alpha1 + 34*inca), *(pi1_lp + 34))
              bli_scopysconvert(*(alpha1 + 35*inca), *(pi1_lp + 35))
              bli_scopysconvert(*(alpha1 + 36*inca), *(pi1_lp + 36))
              bli_scopysconvert(*(alpha1 + 37*inca), *(pi1_lp + 37))
              bli_scopysconvert(*(alpha1 + 38*inca), *(pi1_lp + 38))
              bli_scopysconvert(*(alpha1 + 39*inca), *(pi1_lp + 39))
              bli_scopysconvert(*(alpha1 + 40*inca), *(pi1_lp + 40))
              bli_scopysconvert(*(alpha1 + 41*inca), *(pi1_lp + 41))
              bli_scopysconvert(*(alpha1 + 42*inca), *(pi1_lp + 42))
              bli_scopysconvert(*(alpha1 + 43*inca), *(pi1_lp + 43))
              bli_scopysconvert(*(alpha1 + 44*inca), *(pi1_lp + 44))
              bli_scopysconvert(*(alpha1 + 45*inca), *(pi1_lp + 45))
              bli_scopysconvert(*(alpha1 + 46*inca), *(pi1_lp + 46))
              bli_scopysconvert(*(alpha1 + 47*inca), *(pi1_lp + 47))
              bli_scopysconvert(*(alpha1 + 48*inca), *(pi1_lp + 48))
              bli_scopysconvert(*(alpha1 + 49*inca), *(pi1_lp + 49))
              bli_scopysconvert(*(alpha1 + 50*inca), *(pi1_lp + 50))
              bli_scopysconvert(*(alpha1 + 51*inca), *(pi1_lp + 51))
              bli_scopysconvert(*(alpha1 + 52*inca), *(pi1_lp + 52))
              bli_scopysconvert(*(alpha1 + 53*inca), *(pi1_lp + 53))
              bli_scopysconvert(*(alpha1 + 54*inca), *(pi1_lp + 54))
              bli_scopysconvert(*(alpha1 + 55*inca), *(pi1_lp + 55))
              bli_scopysconvert(*(alpha1 + 56*inca), *(pi1_lp + 56))
              bli_scopysconvert(*(alpha1 + 57*inca), *(pi1_lp + 57))
              bli_scopysconvert(*(alpha1 + 58*inca), *(pi1_lp + 58))
              bli_scopysconvert(*(alpha1 + 59*inca), *(pi1_lp + 59))
              bli_scopysconvert(*(alpha1 + 60*inca), *(pi1_lp + 60))
              bli_scopysconvert(*(alpha1 + 61*inca), *(pi1_lp + 61))
              bli_scopysconvert(*(alpha1 + 62*inca), *(pi1_lp + 62))
              bli_scopysconvert(*(alpha1 + 63*inca), *(pi1_lp + 63))
              bli_scopysconvert(*(alpha1 + 64*inca), *(pi1_lp + 64))
              bli_scopysconvert(*(alpha1 + 65*inca), *(pi1_lp + 65))
              bli_scopysconvert(*(alpha1 + 66*inca), *(pi1_lp + 66))
              bli_scopysconvert(*(alpha1 + 67*inca), *(pi1_lp + 67))
              bli_scopysconvert(*(alpha1 + 68*inca), *(pi1_lp + 68))
              bli_scopysconvert(*(alpha1 + 69*inca), *(pi1_lp + 69))
              bli_scopysconvert(*(alpha1 + 70*inca), *(pi1_lp + 70))
              bli_scopysconvert(*(alpha1 + 71*inca), *(pi1_lp + 71))
              bli_scopysconvert(*(alpha1 + 72*inca), *(pi1_lp + 72))
              bli_scopysconvert(*(alpha1 + 73*inca), *(pi1_lp + 73))
              bli_scopysconvert(*(alpha1 + 74*inca), *(pi1_lp + 74))
              bli_scopysconvert(*(alpha1 + 75*inca), *(pi1_lp + 75))
              bli_scopysconvert(*(alpha1 + 76*inca), *(pi1_lp + 76))
              bli_scopysconvert(*(alpha1 + 77*inca), *(pi1_lp + 77))
              bli_scopysconvert(*(alpha1 + 78*inca), *(pi1_lp + 78))
              bli_scopysconvert(*(alpha1 + 79*inca), *(pi1_lp + 79))
              bli_scopysconvert(*(alpha1 + 80*inca), *(pi1_lp + 80))
              bli_scopysconvert(*(alpha1 + 81*inca), *(pi1_lp + 81))
              bli_scopysconvert(*(alpha1 + 82*inca), *(pi1_lp + 82))
              bli_scopysconvert(*(alpha1 + 83*inca), *(pi1_lp + 83))
              bli_scopysconvert(*(alpha1 + 84*inca), *(pi1_lp + 84))
              bli_scopysconvert(*(alpha1 + 85*inca), *(pi1_lp + 85))
              bli_scopysconvert(*(alpha1 + 86*inca), *(pi1_lp + 86))
              bli_scopysconvert(*(alpha1 + 87*inca), *(pi1_lp + 87))

              alpha1 += lda;
              pi1_lp += ldp;
	    }
	  }
          else
#endif
	  {
            for ( dim_t k = n; k != 0; --k )
            {
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
              bli_scopys(*(alpha1 + 32*inca), *(pi1 + 32))
              bli_scopys(*(alpha1 + 33*inca), *(pi1 + 33))
              bli_scopys(*(alpha1 + 34*inca), *(pi1 + 34))
              bli_scopys(*(alpha1 + 35*inca), *(pi1 + 35))
              bli_scopys(*(alpha1 + 36*inca), *(pi1 + 36))
              bli_scopys(*(alpha1 + 37*inca), *(pi1 + 37))
              bli_scopys(*(alpha1 + 38*inca), *(pi1 + 38))
              bli_scopys(*(alpha1 + 39*inca), *(pi1 + 39))
              bli_scopys(*(alpha1 + 40*inca), *(pi1 + 40))
              bli_scopys(*(alpha1 + 41*inca), *(pi1 + 41))
              bli_scopys(*(alpha1 + 42*inca), *(pi1 + 42))
              bli_scopys(*(alpha1 + 43*inca), *(pi1 + 43))
              bli_scopys(*(alpha1 + 44*inca), *(pi1 + 44))
              bli_scopys(*(alpha1 + 45*inca), *(pi1 + 45))
              bli_scopys(*(alpha1 + 46*inca), *(pi1 + 46))
              bli_scopys(*(alpha1 + 47*inca), *(pi1 + 47))
              bli_scopys(*(alpha1 + 48*inca), *(pi1 + 48))
              bli_scopys(*(alpha1 + 49*inca), *(pi1 + 49))
              bli_scopys(*(alpha1 + 50*inca), *(pi1 + 50))
              bli_scopys(*(alpha1 + 51*inca), *(pi1 + 51))
              bli_scopys(*(alpha1 + 52*inca), *(pi1 + 52))
              bli_scopys(*(alpha1 + 53*inca), *(pi1 + 53))
              bli_scopys(*(alpha1 + 54*inca), *(pi1 + 54))
              bli_scopys(*(alpha1 + 55*inca), *(pi1 + 55))
              bli_scopys(*(alpha1 + 56*inca), *(pi1 + 56))
              bli_scopys(*(alpha1 + 57*inca), *(pi1 + 57))
              bli_scopys(*(alpha1 + 58*inca), *(pi1 + 58))
              bli_scopys(*(alpha1 + 59*inca), *(pi1 + 59))
              bli_scopys(*(alpha1 + 60*inca), *(pi1 + 60))
              bli_scopys(*(alpha1 + 61*inca), *(pi1 + 61))
              bli_scopys(*(alpha1 + 62*inca), *(pi1 + 62))
              bli_scopys(*(alpha1 + 63*inca), *(pi1 + 63))
              bli_scopys(*(alpha1 + 64*inca), *(pi1 + 64))
              bli_scopys(*(alpha1 + 65*inca), *(pi1 + 65))
              bli_scopys(*(alpha1 + 66*inca), *(pi1 + 66))
              bli_scopys(*(alpha1 + 67*inca), *(pi1 + 67))
              bli_scopys(*(alpha1 + 68*inca), *(pi1 + 68))
              bli_scopys(*(alpha1 + 69*inca), *(pi1 + 69))
              bli_scopys(*(alpha1 + 70*inca), *(pi1 + 70))
              bli_scopys(*(alpha1 + 71*inca), *(pi1 + 71))
              bli_scopys(*(alpha1 + 72*inca), *(pi1 + 72))
              bli_scopys(*(alpha1 + 73*inca), *(pi1 + 73))
              bli_scopys(*(alpha1 + 74*inca), *(pi1 + 74))
              bli_scopys(*(alpha1 + 75*inca), *(pi1 + 75))
              bli_scopys(*(alpha1 + 76*inca), *(pi1 + 76))
              bli_scopys(*(alpha1 + 77*inca), *(pi1 + 77))
              bli_scopys(*(alpha1 + 78*inca), *(pi1 + 78))
              bli_scopys(*(alpha1 + 79*inca), *(pi1 + 79))
              bli_scopys(*(alpha1 + 80*inca), *(pi1 + 80))
              bli_scopys(*(alpha1 + 81*inca), *(pi1 + 81))
              bli_scopys(*(alpha1 + 82*inca), *(pi1 + 82))
              bli_scopys(*(alpha1 + 83*inca), *(pi1 + 83))
              bli_scopys(*(alpha1 + 84*inca), *(pi1 + 84))
              bli_scopys(*(alpha1 + 85*inca), *(pi1 + 85))
              bli_scopys(*(alpha1 + 86*inca), *(pi1 + 86))
              bli_scopys(*(alpha1 + 87*inca), *(pi1 + 87))

              alpha1 += lda;
              pi1    += ldp;
	    }
          }
        }
      }
      else //there is a kappa_cast
      {
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
	{
          for ( dim_t k = n; k != 0; --k )
          {
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 0*inca), *(pi1_lp + 0) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 1*inca), *(pi1_lp + 1) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 2*inca), *(pi1_lp + 2) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 3*inca), *(pi1_lp + 3) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 4*inca), *(pi1_lp + 4) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 5*inca), *(pi1_lp + 5) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 6*inca), *(pi1_lp + 6) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 7*inca), *(pi1_lp + 7) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 8*inca), *(pi1_lp + 8) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 9*inca), *(pi1_lp + 9) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 10*inca), *(pi1_lp + 10) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 11*inca), *(pi1_lp + 11) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 12*inca), *(pi1_lp + 12) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 13*inca), *(pi1_lp + 13) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 14*inca), *(pi1_lp + 14) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 15*inca), *(pi1_lp + 15) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 16*inca), *(pi1_lp + 16) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 17*inca), *(pi1_lp + 17) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 18*inca), *(pi1_lp + 18) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 19*inca), *(pi1_lp + 19) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 20*inca), *(pi1_lp + 20) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 21*inca), *(pi1_lp + 21) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 22*inca), *(pi1_lp + 22) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 23*inca), *(pi1_lp + 23) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 24*inca), *(pi1_lp + 24) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 25*inca), *(pi1_lp + 25) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 26*inca), *(pi1_lp + 26) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 27*inca), *(pi1_lp + 27) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 28*inca), *(pi1_lp + 28) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 29*inca), *(pi1_lp + 29) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 30*inca), *(pi1_lp + 30) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 31*inca), *(pi1_lp + 31) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 32*inca), *(pi1_lp + 32) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 33*inca), *(pi1_lp + 33) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 34*inca), *(pi1_lp + 34) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 35*inca), *(pi1_lp + 35) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 36*inca), *(pi1_lp + 36) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 37*inca), *(pi1_lp + 37) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 38*inca), *(pi1_lp + 38) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 39*inca), *(pi1_lp + 39) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 40*inca), *(pi1_lp + 40) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 41*inca), *(pi1_lp + 41) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 42*inca), *(pi1_lp + 42) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 43*inca), *(pi1_lp + 43) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 44*inca), *(pi1_lp + 44) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 45*inca), *(pi1_lp + 45) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 46*inca), *(pi1_lp + 46) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 47*inca), *(pi1_lp + 47) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 48*inca), *(pi1_lp + 48) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 49*inca), *(pi1_lp + 49) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 50*inca), *(pi1_lp + 50) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 51*inca), *(pi1_lp + 51) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 52*inca), *(pi1_lp + 52) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 53*inca), *(pi1_lp + 53) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 54*inca), *(pi1_lp + 54) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 55*inca), *(pi1_lp + 55) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 56*inca), *(pi1_lp + 56) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 57*inca), *(pi1_lp + 57) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 58*inca), *(pi1_lp + 58) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 59*inca), *(pi1_lp + 59) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 60*inca), *(pi1_lp + 60) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 61*inca), *(pi1_lp + 61) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 62*inca), *(pi1_lp + 62) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 63*inca), *(pi1_lp + 63) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 64*inca), *(pi1_lp + 64) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 65*inca), *(pi1_lp + 65) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 66*inca), *(pi1_lp + 66) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 67*inca), *(pi1_lp + 67) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 68*inca), *(pi1_lp + 68) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 69*inca), *(pi1_lp + 69) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 70*inca), *(pi1_lp + 70) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 71*inca), *(pi1_lp + 71) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 72*inca), *(pi1_lp + 72) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 73*inca), *(pi1_lp + 73) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 74*inca), *(pi1_lp + 74) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 75*inca), *(pi1_lp + 75) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 76*inca), *(pi1_lp + 76) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 77*inca), *(pi1_lp + 77) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 78*inca), *(pi1_lp + 78) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 79*inca), *(pi1_lp + 79) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 80*inca), *(pi1_lp + 80) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 81*inca), *(pi1_lp + 81) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 82*inca), *(pi1_lp + 82) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 83*inca), *(pi1_lp + 83) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 84*inca), *(pi1_lp + 84) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 85*inca), *(pi1_lp + 85) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 86*inca), *(pi1_lp + 86) );
            bli_sscal2sconvert( *kappa_cast, *(alpha1 + 87*inca), *(pi1_lp + 87) );

            alpha1 += lda;
            pi1_lp    += ldp;
          }
        }
        else
#endif
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
            bli_sscal2s( *kappa_cast, *(alpha1 + 32*inca), *(pi1 + 32) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 33*inca), *(pi1 + 33) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 34*inca), *(pi1 + 34) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 35*inca), *(pi1 + 35) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 36*inca), *(pi1 + 36) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 37*inca), *(pi1 + 37) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 38*inca), *(pi1 + 38) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 39*inca), *(pi1 + 39) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 40*inca), *(pi1 + 40) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 41*inca), *(pi1 + 41) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 42*inca), *(pi1 + 42) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 43*inca), *(pi1 + 43) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 44*inca), *(pi1 + 44) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 45*inca), *(pi1 + 45) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 46*inca), *(pi1 + 46) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 47*inca), *(pi1 + 47) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 48*inca), *(pi1 + 48) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 49*inca), *(pi1 + 49) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 50*inca), *(pi1 + 50) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 51*inca), *(pi1 + 51) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 52*inca), *(pi1 + 52) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 53*inca), *(pi1 + 53) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 54*inca), *(pi1 + 54) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 55*inca), *(pi1 + 55) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 56*inca), *(pi1 + 56) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 57*inca), *(pi1 + 57) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 58*inca), *(pi1 + 58) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 59*inca), *(pi1 + 59) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 60*inca), *(pi1 + 60) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 61*inca), *(pi1 + 61) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 62*inca), *(pi1 + 62) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 63*inca), *(pi1 + 63) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 64*inca), *(pi1 + 64) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 65*inca), *(pi1 + 65) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 66*inca), *(pi1 + 66) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 67*inca), *(pi1 + 67) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 68*inca), *(pi1 + 68) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 69*inca), *(pi1 + 69) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 70*inca), *(pi1 + 70) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 71*inca), *(pi1 + 71) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 72*inca), *(pi1 + 72) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 73*inca), *(pi1 + 73) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 74*inca), *(pi1 + 74) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 75*inca), *(pi1 + 75) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 76*inca), *(pi1 + 76) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 77*inca), *(pi1 + 77) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 78*inca), *(pi1 + 78) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 79*inca), *(pi1 + 79) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 80*inca), *(pi1 + 80) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 81*inca), *(pi1 + 81) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 82*inca), *(pi1 + 82) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 83*inca), *(pi1 + 83) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 84*inca), *(pi1 + 84) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 85*inca), *(pi1 + 85) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 86*inca), *(pi1 + 86) );
            bli_sscal2s( *kappa_cast, *(alpha1 + 87*inca), *(pi1 + 87) );

            alpha1 += lda;
            pi1    += ldp;
          }
        }
      }
    }
    else // cdim < mnr
    // if the matrix size is smaller than the panel size
    // then copy as-is and scale by kappa (not need to pack)
    {

#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < cdim; i++)
            {
              bli_sscal2sconvert( *kappa_cast, *(alpha1 + i*inca), *(pi1_lp + i) );
            }

            alpha1 += lda;
            pi1_lp += ldp;
          }
        }
        else
#endif
        {
          for ( dim_t k = n; k != 0; --k )
          {
            for ( int i = 0; i < cdim; i++)
            {
              bli_sscal2s( *kappa_cast, *(alpha1 + i*inca), *(pi1 + i) );
            }

            alpha1 += lda;
            pi1    += ldp;
          }
        }

      // pad with zeros if the panel size is greater than the matrix size

      const dim_t     i      = cdim;
      const dim_t     m_edge = mnr - cdim;
      const dim_t     n_edge = n_max;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (i  )*1;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (i  )*1;
        for ( dim_t jj = 0; jj < n_edge; ++jj ) {
        for ( dim_t ii = 0; ii < m_edge; ++ii ) {
          *(p_edge_lp + ii + jj*ldp) = 0;
        }}
      }
      else
#endif
      {
        bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
      }
    }

    // pad with zeros if there is a difference between the logical size and physical size
    if ( n < n_max )
    {
      const dim_t     j      = n;
      const dim_t     m_edge = mnr;
      const dim_t     n_edge = n_max - n;
      float* restrict p_cast = p;
      float* restrict p_edge = p_cast + (j  )*ldp;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (j  )*ldp;
        for ( dim_t jj = 0; jj < n_edge; ++jj )
        for ( dim_t ii = 0; ii < m_edge; ++ii )
          *(p_edge_lp + ii + jj*ldp) = 0;
      }
      else
#endif
      {
        bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
      }
    }

}
