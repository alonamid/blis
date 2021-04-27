/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, The University of Texas at Austin

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

// -- level-1v - CPU --
INVERTV_KER_PROT( float, s, invertv_lowprec)
SETV_KER_PROT( float, s, setv_lowprec)
SCALV_KER_PROT( float, s, scalv_lowprec)
SCAL2V_KER_PROT( float, s, scal2v_lowprec)
COPYV_KER_PROT( float, s, copyv_lowprec)

// -- level-1v - Hwacha --
ADDV_KER_PROT( float,   s, addv_hwacha )
AXPYV_KER_PROT( float,   s, axpyv_hwacha )
XPBYV_KER_PROT( float,   s, xpbyv_hwacha )
AXPBYV_KER_PROT( float,   s, axpbyv_hwacha )
SUBV_KER_PROT( float,   s, subv_hwacha )
SWAPV_KER_PROT( float,   s, swapv_hwacha )
COPYV_KER_PROT( float,   s, copyv_hwacha )
SETV_KER_PROT( float,   s, setv_hwacha )
SCALV_KER_PROT( float,   s, scalv_hwacha )
SCAL2V_KER_PROT( float,   s, scal2v_hwacha )
INVERTV_KER_PROT( float,   s, invertv_hwacha )
DOTV_KER_PROT( float,   s, dotv_hwacha )
DOTXV_KER_PROT( float,   s, dotxv_hwacha )

// -- level-1f --
DOTXF_KER_PROT( float,   s, dotxf_hwacha )
AXPYF_KER_PROT( float,   s, axpyf_hwacha )
AXPY2V_KER_PROT( float,   s, axpy2v_hwacha )
DOTAXPYV_KER_PROT( float,   s, dotaxpyv_hwacha )

// -- packing --
PACKM_KER_PROT( float,   s, packm_gemmini_cxk )
PACKM_KER_PROT( float,   s, packm_gemmini_88xk )
PACKM_KER_PROT( float,   s, packm_gemmini_32xk )
PACKM_KER_PROT( float,   s, packm_gemmini_4xk )
PACKM_KER_PROT( float,   s, packm_hwacha_cxk )

// -- level-3 --

// gemmini
TRSM_UKR_PROT( float,   s, trsm_u_gemmini_small )
TRSM_UKR_PROT( float,   s, trsm_l_gemmini_small )
GEMM_UKR_PROT( float,   s, gemm_gemmini_fsm_ws )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_l_gemmini_fsm_ws )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_u_gemmini_fsm_ws )
GEMM_UKR_PROT( float,   s, gemm_gemmini_small_os )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_u_gemmini_small_os )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_l_gemmini_small_os )
GEMM_UKR_PROT( float,   s, gemm_gemmini_small_ws )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_l_gemmini_small_ws )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_u_gemmini_small_ws )

// hwacha
GEMM_UKR_PROT( float,   s, gemm_hwacha_16xn )
TRSM_UKR_PROT( float,   s, trsm_l_hwacha_16xn )
TRSM_UKR_PROT( float,   s, trsm_u_hwacha_16xn )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_l_hwacha_16xn )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_u_hwacha_16xn )

