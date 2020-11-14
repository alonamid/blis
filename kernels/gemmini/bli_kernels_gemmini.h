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

// -- l1 --
INVERTV_KER_PROT( float, s, invertv_lowprec)
SETV_KER_PROT( float, s, setv_lowprec)
SCALV_KER_PROT( float, s, scalv_lowprec)
SCAL2V_KER_PROT( float, s, scal2v_lowprec)
COPYV_KER_PROT( float, s, copyv_lowprec)

// -- packing --
PACKM_KER_PROT( float,   s, packm_gemmini_32xk )
PACKM_KER_PROT( float,   s, packm_gemmini_4xk )
PACKM_KER_PROT( float,   s, packm_gemmini_cxk )

// -- level-3 --

// gemm (asm d12x6)
TRSM_UKR_PROT( float,   s, trsm_u_gemmini_small )
TRSM_UKR_PROT( float,   s, trsm_l_gemmini_small )
GEMM_UKR_PROT( float,   s, gemm_gemmini_small_os )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_u_gemmini_small_os )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_l_gemmini_small_os )
GEMM_UKR_PROT( float,   s, gemm_gemmini_small_ws )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_u_gemmini_small_ws )
GEMMTRSM_UKR_PROT( float,   s, gemmtrsm_l_gemmini_small_ws )
