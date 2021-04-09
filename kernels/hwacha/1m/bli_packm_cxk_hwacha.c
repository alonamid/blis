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

extern void bli_packm_hwacha_vf_init(void) __attribute__((visibility("protected")));
extern void bli_packm_hwacha_vf_sset0(void) __attribute__((visibility("protected")));
extern void bli_packm_hwacha_vf_scopy(void) __attribute__((visibility("protected")));
extern void bli_packm_hwacha_vf_scopyconvert(void) __attribute__((visibility("protected")));
extern void bli_packm_hwacha_vf_sscal2(void) __attribute__((visibility("protected")));
extern void bli_packm_hwacha_vf_sscal2convert(void) __attribute__((visibility("protected")));

//generic, but not unrolled
//should be functional, but lower-perf
void bli_spackm_hwacha_cxk
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

    //TODO: this should be in the blis context initialization
    __asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 2, 1, 1)));
    int vlen_result;

    float*  restrict kappa_cast = kappa;
    float*  restrict alpha1     = a;
    float*  restrict pi1        = p;
    elem_t* restrict pi1_lp     = (elem_t*)p;

    dim_t           mnr        = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_MR, cntx );;


    if ( cdim == mnr ) //the "standard" case for packing, where a big matrix needs to be packed into panels
    {
      __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (mnr));
      vf(bli_packm_hwacha_vf_init);
      if (*kappa_cast == 0) // no kappa_cast
      {
	if ( bli_is_conj( conja ) )
        {
          for (dim_t k = n; k != 0; --k)
          {
            MEMTOUCH(pi1, float, vlen_result);
            __asm__ volatile ("vmca va0,  %0" : : "r" (pi1));
            __asm__ volatile ("vmca va1,  %0" : : "r" (alpha1));
            __asm__ volatile ("vmca va2,  %0" : : "r" (inca*sizeof(float)));
            vf(bli_packm_hwacha_vf_scopy);
            alpha1 += lda;
            pi1    += ldp;
          }
        }
        else
        {
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	  if (bli_cntx_lowprec_in_use(cntx))
	  {
            for (dim_t k = n; k != 0; --k)
            {
              MEMTOUCH(pi1_lp, float, vlen_result);
              __asm__ volatile ("vmca va0,  %0" : : "r" (pi1_lp));
              __asm__ volatile ("vmca va1,  %0" : : "r" (alpha1));
              __asm__ volatile ("vmca va2,  %0" : : "r" (inca*sizeof(float)));
              vf(bli_packm_hwacha_vf_scopyconvert);
              alpha1 += lda;
              pi1_lp += ldp;
            }
          }
          else
#endif
          {
            for (dim_t k = n; k != 0; --k)
            {
              MEMTOUCH(pi1, float, vlen_result);
              __asm__ volatile ("vmca va0,  %0" : : "r" (pi1));
              __asm__ volatile ("vmca va1,  %0" : : "r" (alpha1));
              __asm__ volatile ("vmca va2,  %0" : : "r" (inca*sizeof(float)));
              vf(bli_packm_hwacha_vf_scopy);
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
          for (dim_t k = n; k != 0; --k)
          {
            MEMTOUCH(pi1_lp, float, vlen_result);
            __asm__ volatile ("vmca va0,  %0" : : "r" (pi1_lp));
            __asm__ volatile ("vmca va1,  %0" : : "r" (alpha1));
            __asm__ volatile ("vmca va2,  %0" : : "r" (inca*sizeof(float)));
            __asm__ volatile ("vmcs vs1,  %0" : : "r" (*kappa_cast));
            vf(bli_packm_hwacha_vf_sscal2convert);
            alpha1 += lda;
            pi1_lp += ldp;
          }
        }
        else
#endif
        {
          for (dim_t k = n; k != 0; --k)
          {
            MEMTOUCH(pi1, float, vlen_result);
            __asm__ volatile ("vmca va0,  %0" : : "r" (pi1));
            __asm__ volatile ("vmca va1,  %0" : : "r" (alpha1));
            __asm__ volatile ("vmca va2,  %0" : : "r" (inca*sizeof(float)));
            __asm__ volatile ("vmcs vs1,  %0" : : "r" (*kappa_cast));
            vf(bli_packm_hwacha_vf_sscal2);
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
        __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (cdim));
        vf(bli_packm_hwacha_vf_init);
#ifdef ELEM_T_IS_LOWPREC_FLOAT
	if (bli_cntx_lowprec_in_use(cntx))
        {
          for (dim_t k = n; k != 0; --k)
          {
            MEMTOUCH(pi1_lp, float, vlen_result);
            __asm__ volatile ("vmca va0,  %0" : : "r" (pi1_lp));
            __asm__ volatile ("vmca va1,  %0" : : "r" (alpha1));
            __asm__ volatile ("vmca va2,  %0" : : "r" (inca*sizeof(float)));
            __asm__ volatile ("vmcs vs1,  %0" : : "r" (*kappa_cast));
            vf(bli_packm_hwacha_vf_sscal2convert);
            alpha1 += lda;
            pi1_lp += ldp;
          }
        }
        else
#endif
        {
          for (dim_t k = n; k != 0; --k)
          {
            MEMTOUCH(pi1, float, vlen_result);
            __asm__ volatile ("vmca va0,  %0" : : "r" (pi1));
            __asm__ volatile ("vmca va1,  %0" : : "r" (alpha1));
            __asm__ volatile ("vmca va2,  %0" : : "r" (inca*sizeof(float)));
            __asm__ volatile ("vmcs vs1,  %0" : : "r" (*kappa_cast));
            vf(bli_packm_hwacha_vf_sscal2);
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

      __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (m_edge));
      vf(bli_packm_hwacha_vf_init);

#ifdef ELEM_T_IS_LOWPREC_FLOAT
      if (bli_cntx_lowprec_in_use(cntx))
      {
        elem_t* restrict p_edge_lp = (elem_t*)p_cast + (i  )*1;
        for ( dim_t jj = 0; jj < n_edge; ++jj ) {
          __asm__ volatile ("vmca va0,  %0" : : "r" (p_edge_lp + jj*ldp));
          MEMTOUCH(p_edge_lp + jj*ldp, float, vlen_result);
          vf(bli_packm_hwacha_vf_sset0);
        }
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

    __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (m_edge));
    vf(bli_packm_hwacha_vf_init);

#ifdef ELEM_T_IS_LOWPREC_FLOAT
    if (bli_cntx_lowprec_in_use(cntx))
    {
      elem_t* restrict p_edge_lp = (elem_t*)p_cast + (j  )*ldp;
      for ( dim_t jj = 0; jj < n_edge; ++jj ) {
        __asm__ volatile ("vmca va0,  %0" : : "r" (p_edge_lp + jj*ldp));
        MEMTOUCH(p_edge_lp + jj*ldp, float, vlen_result);
        vf(bli_packm_hwacha_vf_sset0);
      }
    }
    else
#endif
    {
      bli_sset0s_mxn( m_edge, n_edge, p_edge, 1, ldp);
    }
  }

  __asm__ volatile ("fence" ::: "memory");
}
