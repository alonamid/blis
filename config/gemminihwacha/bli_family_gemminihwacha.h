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

//#ifndef BLIS_FAMILY_H
//#define BLIS_FAMILY_H
#include "include/gemmini_params.h"

// remove to disable VRU
#define VRU_ENABLE

#ifdef VRU_ENABLE
// because gcc complains about shifting without L
#define VRU_SWITCH 0x8000000000000000
#else
#define VRU_SWITCH 0x0
#endif


#define VCFG(nvvd, nvvw, nvvh, nvp) \
  (((nvvd) & 0x1ff) | \
  (((nvp) & 0x1f) << 9) | \
  (((nvvw) & 0x1ff) << 14) | \
  (((nvvh) & 0x1ff) << 23) | \
  (VRU_SWITCH))

#define vf(p) \
        __asm__ __volatile__ ("vf (%0)" : : "r" (p))

#define HWACHA_MIN_DIM 8
#define SMAXVL 2048

#define BLIS_ENABLE_SMALL_MATRIX_TRSM
#define BLIS_SMALL_MATRIX_THRES_TRSM DIM*DIM

#define BLIS_STACK_BUF_MAX_SIZE  ( ACC_ROWS * MAX_BYTES * 2 )

#define MEMTOUCH(iaddr, type, bound) ({                           \
      volatile type* addr = iaddr;                                \
      volatile type t;                                            \
      t = (addr)[0];                                              \
      (addr)[0] = t;                                              \
      volatile type* tf = (type*) (((((uintptr_t) (addr)) >> 12) + 1) << 12);    \
      for (; tf - (addr) < bound; tf += (1 << 12) / sizeof(type)) {     \
        t = tf[0];                                                      \
        tf[0] = t;                                                      \
      }                                                                 \
      __asm__ __volatile__ ("fence" ::: "memory"); \
    })


//#endif

