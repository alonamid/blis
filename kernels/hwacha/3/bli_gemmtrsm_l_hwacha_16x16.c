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


extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_0(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_1(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_2(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_3(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_4(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_5(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_6(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_7(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_8(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_9(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_10(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_11(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_12(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_13(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_14(void) __attribute__((visibility("protected")));
extern void bli_sgemmtrsm_l_hwacha_16xn_vf_inner_15(void) __attribute__((visibility("protected")));

extern void bli_sgemm_hwacha_16xn_vf_init_beta(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_tail(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_inner_0(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_inner_1(void) __attribute__((visibility("protected")));

#define vf(p) \
        __asm__ __volatile__ ("vf (%0)" : : "r" (p))


void bli_sgemmtrsm_l_hwacha_16x16
     (
       dim_t               k,
       float*    restrict alpha,
       float*    restrict a10,
       float*    restrict a11,
       float*    restrict b01,
       float*    restrict b11,
       float*    restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
/*
  Template trsm_l micro-kernel implementation

  This function contains a template implementation for a double-precision
  complex trsm micro-kernel, coded in C, which can serve as the starting point
  for one to write an optimized micro-kernel on an arbitrary architecture.
  (We show a template implementation for only double-precision complex because
  the templates for the other three floating-point types would be nearly
  identical.)

  This micro-kernel performs the following operation:

    C11 := inv(A11) * B11

  where A11 is MR x MR and lower triangular, B11 is MR x NR, and C11 is
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

        float* restrict minus_one = bli_sm1;


        //TODO: this should be in the blis context initialization
        __asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, mr+2, 0, 1)));

        int vlen_result;
        __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (nr));
        if (vlen_result < nr)
        {
          printf("ERROR: vlen=%d is smaller than NR=%ld\n", vlen_result, nr);
          exit(-1);
        }

        float* a_ptr = a10;
        float* b_ptr = b01;

        /* vv0-15 are B11 rows */
        // B11 rows addresses
        __asm__ volatile ("vmca va0,  %0" : : "r" (b11+0*rs_b));
        __asm__ volatile ("vmca va1,  %0" : : "r" (b11+1*rs_b));
        __asm__ volatile ("vmca va2,  %0" : : "r" (b11+2*rs_b));
        __asm__ volatile ("vmca va3,  %0" : : "r" (b11+3*rs_b));
        __asm__ volatile ("vmca va4,  %0" : : "r" (b11+4*rs_b));
        __asm__ volatile ("vmca va5,  %0" : : "r" (b11+5*rs_b));
        __asm__ volatile ("vmca va6,  %0" : : "r" (b11+6*rs_b));
        __asm__ volatile ("vmca va7,  %0" : : "r" (b11+7*rs_b));
        __asm__ volatile ("vmca va8,  %0" : : "r" (b11+8*rs_b));
        __asm__ volatile ("vmca va9,  %0" : : "r" (b11+9*rs_b));
        __asm__ volatile ("vmca va10, %0" : : "r" (b11+10*rs_b));
        __asm__ volatile ("vmca va11, %0" : : "r" (b11+11*rs_b));
        __asm__ volatile ("vmca va12, %0" : : "r" (b11+12*rs_b));
        __asm__ volatile ("vmca va13, %0" : : "r" (b11+13*rs_b));
        __asm__ volatile ("vmca va14, %0" : : "r" (b11+14*rs_b));
        __asm__ volatile ("vmca va15, %0" : : "r" (b11+15*rs_b));

        //B01 row address
        __asm__ volatile ("vmca va16, %0 \n\t" : : "r" (b_ptr));
        b_ptr += rs_b;

        // load B11 and first B01
        __asm__ volatile ("vmcs vs63,  %0" : : "r" (*alpha));
	vf(&bli_sgemm_hwacha_16xn_vf_init_beta);
    
        // load alpha
        __asm__ volatile ("vmcs vs63,  %0" : : "r" (*minus_one));


        for (; k > 1; k-=2)
        {
          // A column
          __asm__ volatile ("vmcs vs1, %0 \n\t"
                          "vmcs vs2, %1 \n\t"
                          "vmcs vs3, %2 \n\t"
                          "vmcs vs4, %3 \n\t"
        
                          "vmcs vs5, %4 \n\t"
                          "vmcs vs6, %5 \n\t"
                          "vmcs vs7, %6 \n\t"
                          "vmcs vs8, %7 \n\t"
        
                          "vmcs vs9, %8 \n\t"
                          "vmcs vs10, %9 \n\t"
                          "vmcs vs11, %10\n\t"
                          "vmcs vs12, %11\n\t"
        
                          "vmcs vs13, %12\n\t"
                          "vmcs vs14, %13\n\t"
                          "vmcs vs15, %14\n\t"
                          "vmcs vs16, %15\n\t"
                      : 
                      : "r" (*(a_ptr+0)),  "r" (*(a_ptr+1)),  "r" (*(a_ptr+2)),  "r" (*(a_ptr+3)), 
                        "r" (*(a_ptr+4)),  "r" (*(a_ptr+5)),  "r" (*(a_ptr+6)),  "r" (*(a_ptr+7)), 
                        "r" (*(a_ptr+8)),  "r" (*(a_ptr+9)),  "r" (*(a_ptr+10)), "r" (*(a_ptr+11)), 
                        "r" (*(a_ptr+12)), "r" (*(a_ptr+13)), "r" (*(a_ptr+14)), "r" (*(a_ptr+15)) 
                    );
            
          // B row
          __asm__ volatile ("vmca va16, %0 \n\t" : : "r" (b_ptr));

          vf(&bli_sgemm_hwacha_16xn_vf_inner_0);

          b_ptr += rs_b;

          // A column
          __asm__ volatile ("vmcs vs1, %0 \n\t"
                          "vmcs vs2, %1 \n\t"
                          "vmcs vs3, %2 \n\t"
                          "vmcs vs4, %3 \n\t"
        
                          "vmcs vs5, %4 \n\t"
                          "vmcs vs6, %5 \n\t"
                          "vmcs vs7, %6 \n\t"
                          "vmcs vs8, %7 \n\t"
        
                          "vmcs vs9, %8 \n\t"
                          "vmcs vs10, %9 \n\t"
                          "vmcs vs11, %10\n\t"
                          "vmcs vs12, %11\n\t"
        
                          "vmcs vs13, %12\n\t"
                          "vmcs vs14, %13\n\t"
                          "vmcs vs15, %14\n\t"
                          "vmcs vs16, %15\n\t"
                      : 
                      : "r" (*(a_ptr+16)),  "r" (*(a_ptr+17)),  "r" (*(a_ptr+18)),  "r" (*(a_ptr+19)), 
                        "r" (*(a_ptr+20)),  "r" (*(a_ptr+21)),  "r" (*(a_ptr+22)),  "r" (*(a_ptr+23)), 
                        "r" (*(a_ptr+24)),  "r" (*(a_ptr+25)),  "r" (*(a_ptr+26)), "r" (*(a_ptr+27)), 
                        "r" (*(a_ptr+28)), "r" (*(a_ptr+29)), "r" (*(a_ptr+30)), "r" (*(a_ptr+31)) 
                    );

          // B row
          __asm__ volatile ("vmca va16, %0 \n\t" : : "r" (b_ptr));

          vf(&bli_sgemm_hwacha_16xn_vf_inner_1);

          b_ptr += rs_b;
          a_ptr += 2 * mr;

        }

        if (k > 0)
        {
          // A column
          __asm__ volatile ("vmcs vs1, %0 \n\t"
                          "vmcs vs2, %1 \n\t"
                          "vmcs vs3, %2 \n\t"
                          "vmcs vs4, %3 \n\t"
        
                          "vmcs vs5, %4 \n\t"
                          "vmcs vs6, %5 \n\t"
                          "vmcs vs7, %6 \n\t"
                          "vmcs vs8, %7 \n\t"
        
                          "vmcs vs9, %8 \n\t"
                          "vmcs vs10, %9 \n\t"
                          "vmcs vs11, %10\n\t"
                          "vmcs vs12, %11\n\t"
        
                          "vmcs vs13, %12\n\t"
                          "vmcs vs14, %13\n\t"
                          "vmcs vs15, %14\n\t"
                          "vmcs vs16, %15\n\t"
                      : 
                      : "r" (*(a_ptr+0)),  "r" (*(a_ptr+1)),  "r" (*(a_ptr+2)),  "r" (*(a_ptr+3)), 
                        "r" (*(a_ptr+4)),  "r" (*(a_ptr+5)),  "r" (*(a_ptr+6)),  "r" (*(a_ptr+7)), 
                        "r" (*(a_ptr+8)),  "r" (*(a_ptr+9)),  "r" (*(a_ptr+10)), "r" (*(a_ptr+11)), 
                        "r" (*(a_ptr+12)), "r" (*(a_ptr+13)), "r" (*(a_ptr+14)), "r" (*(a_ptr+15)) 
                    );

          vf(&bli_sgemm_hwacha_16xn_vf_tail);
        }

        /* vv16-31 are C11 rows */
        // C11 rows addresses
        __asm__ volatile ("vmca va16,  %0" : : "r" (c11+0*rs_c));
        __asm__ volatile ("vmca va17,  %0" : : "r" (c11+1*rs_c));
        __asm__ volatile ("vmca va18,  %0" : : "r" (c11+2*rs_c));
        __asm__ volatile ("vmca va19,  %0" : : "r" (c11+3*rs_c));
        __asm__ volatile ("vmca va20,  %0" : : "r" (c11+4*rs_c));
        __asm__ volatile ("vmca va21,  %0" : : "r" (c11+5*rs_c));
        __asm__ volatile ("vmca va22,  %0" : : "r" (c11+6*rs_c));
        __asm__ volatile ("vmca va23,  %0" : : "r" (c11+7*rs_c));
        __asm__ volatile ("vmca va24,  %0" : : "r" (c11+8*rs_c));
        __asm__ volatile ("vmca va25,  %0" : : "r" (c11+9*rs_c));
        __asm__ volatile ("vmca va26, %0" : : "r" (c11+10*rs_c));
        __asm__ volatile ("vmca va27, %0" : : "r" (c11+11*rs_c));
        __asm__ volatile ("vmca va28, %0" : : "r" (c11+12*rs_c));
        __asm__ volatile ("vmca va29, %0" : : "r" (c11+13*rs_c));
        __asm__ volatile ("vmca va30, %0" : : "r" (c11+14*rs_c));
        __asm__ volatile ("vmca va31, %0" : : "r" (c11+15*rs_c));

        /* reference: http://developer.amd.com/wordpress/media/2013/12/Optimization-of-BLIS-Library-for-AMD-ZEN.pdf */

        /* b_0 */
        /* alpha_00 */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_0);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (1  )*rs_a + (1  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (1  )*rs_a + (0  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_1);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (2  )*rs_a + (2  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (2  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (2  )*rs_a + (1  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_2);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (3  )*rs_a + (3  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (3  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (3  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (3  )*rs_a + (2  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_3);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (4  )*rs_a + (4  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (4  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (4  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (4  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 + (4  )*rs_a + (3  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_4);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (5  )*rs_a + (5  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (5  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (5  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (5  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 + (5  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 + (5  )*rs_a + (4  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_5);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (6  )*rs_a + (6  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (6  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (6  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (6  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 + (6  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 + (6  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 + (6  )*rs_a + (5  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_6);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (7  )*rs_a + (7  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (7  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (7  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (7  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 + (7  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 + (7  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 + (7  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 + (7  )*rs_a + (6  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_7);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (8  )*rs_a + (8  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (8  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (8  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (8  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 + (8  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 + (8  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 + (8  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 + (8  )*rs_a + (6  )*cs_a)));
        __asm__ volatile ("vmcs vs9,  %0" : : "r" (*(a11 + (8  )*rs_a + (7  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_8);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 +  (9  )*rs_a + (9  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 +  (9  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 +  (9  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 +  (9  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 +  (9  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 +  (9  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 +  (9  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 +  (9  )*rs_a + (6  )*cs_a)));
        __asm__ volatile ("vmcs vs9,  %0" : : "r" (*(a11 +  (9  )*rs_a + (7  )*cs_a)));
        __asm__ volatile ("vmcs vs10,  %0" : : "r" (*(a11 + (9  )*rs_a + (8  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_9);
       
        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 +  (10  )*rs_a + (10  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 +  (10  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 +  (10  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 +  (10  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 +  (10  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 +  (10  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 +  (10  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 +  (10  )*rs_a + (6  )*cs_a)));
        __asm__ volatile ("vmcs vs9,  %0" : : "r" (*(a11 +  (10  )*rs_a + (7  )*cs_a)));
        __asm__ volatile ("vmcs vs10,  %0" : : "r" (*(a11 + (10  )*rs_a + (8  )*cs_a)));
        __asm__ volatile ("vmcs vs11,  %0" : : "r" (*(a11 + (10  )*rs_a + (9  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_10);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 +  (11  )*rs_a + (11  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 +  (11  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 +  (11  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 +  (11  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 +  (11  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 +  (11  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 +  (11  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 +  (11  )*rs_a + (6  )*cs_a)));
        __asm__ volatile ("vmcs vs9,  %0" : : "r" (*(a11 +  (11  )*rs_a + (7  )*cs_a)));
        __asm__ volatile ("vmcs vs10,  %0" : : "r" (*(a11 + (11  )*rs_a + (8  )*cs_a)));
        __asm__ volatile ("vmcs vs11,  %0" : : "r" (*(a11 + (11  )*rs_a + (9  )*cs_a)));
        __asm__ volatile ("vmcs vs12,  %0" : : "r" (*(a11 + (11  )*rs_a + (10  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_11);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 +  (12  )*rs_a + (12  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 +  (12  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 +  (12  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 +  (12  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 +  (12  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 +  (12  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 +  (12  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 +  (12  )*rs_a + (6  )*cs_a)));
        __asm__ volatile ("vmcs vs9,  %0" : : "r" (*(a11 +  (12  )*rs_a + (7  )*cs_a)));
        __asm__ volatile ("vmcs vs10,  %0" : : "r" (*(a11 + (12  )*rs_a + (8  )*cs_a)));
        __asm__ volatile ("vmcs vs11,  %0" : : "r" (*(a11 + (12  )*rs_a + (9  )*cs_a)));
        __asm__ volatile ("vmcs vs12,  %0" : : "r" (*(a11 + (12  )*rs_a + (10  )*cs_a)));
        __asm__ volatile ("vmcs vs13,  %0" : : "r" (*(a11 + (12  )*rs_a + (11  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_12);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 +  (13  )*rs_a + (13  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 +  (13  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 +  (13  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 +  (13  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 +  (13  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 +  (13  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 +  (13  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 +  (13  )*rs_a + (6  )*cs_a)));
        __asm__ volatile ("vmcs vs9,  %0" : : "r" (*(a11 +  (13  )*rs_a + (7  )*cs_a)));
        __asm__ volatile ("vmcs vs10,  %0" : : "r" (*(a11 + (13  )*rs_a + (8  )*cs_a)));
        __asm__ volatile ("vmcs vs11,  %0" : : "r" (*(a11 + (13  )*rs_a + (9  )*cs_a)));
        __asm__ volatile ("vmcs vs12,  %0" : : "r" (*(a11 + (13  )*rs_a + (10  )*cs_a)));
        __asm__ volatile ("vmcs vs13,  %0" : : "r" (*(a11 + (13  )*rs_a + (11  )*cs_a)));
        __asm__ volatile ("vmcs vs14,  %0" : : "r" (*(a11 + (13  )*rs_a + (12  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_13);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 +  (14  )*rs_a + (14  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 +  (14  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 +  (14  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 +  (14  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 +  (14  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 +  (14  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 +  (14  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 +  (14  )*rs_a + (6  )*cs_a)));
        __asm__ volatile ("vmcs vs9,  %0" : : "r" (*(a11 +  (14  )*rs_a + (7  )*cs_a)));
        __asm__ volatile ("vmcs vs10,  %0" : : "r" (*(a11 + (14  )*rs_a + (8  )*cs_a)));
        __asm__ volatile ("vmcs vs11,  %0" : : "r" (*(a11 + (14  )*rs_a + (9  )*cs_a)));
        __asm__ volatile ("vmcs vs12,  %0" : : "r" (*(a11 + (14  )*rs_a + (10  )*cs_a)));
        __asm__ volatile ("vmcs vs13,  %0" : : "r" (*(a11 + (14  )*rs_a + (11  )*cs_a)));
        __asm__ volatile ("vmcs vs14,  %0" : : "r" (*(a11 + (14  )*rs_a + (12  )*cs_a)));
        __asm__ volatile ("vmcs vs15,  %0" : : "r" (*(a11 + (14  )*rs_a + (13  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_14);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 +  (15  )*rs_a + (15  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 +  (15  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 +  (15  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 +  (15  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 +  (15  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 +  (15  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 +  (15  )*rs_a + (5  )*cs_a)));
        __asm__ volatile ("vmcs vs8,  %0" : : "r" (*(a11 +  (15  )*rs_a + (6  )*cs_a)));
        __asm__ volatile ("vmcs vs9,  %0" : : "r" (*(a11 +  (15  )*rs_a + (7  )*cs_a)));
        __asm__ volatile ("vmcs vs10,  %0" : : "r" (*(a11 + (15  )*rs_a + (8  )*cs_a)));
        __asm__ volatile ("vmcs vs11,  %0" : : "r" (*(a11 + (15  )*rs_a + (9  )*cs_a)));
        __asm__ volatile ("vmcs vs12,  %0" : : "r" (*(a11 + (15  )*rs_a + (10  )*cs_a)));
        __asm__ volatile ("vmcs vs13,  %0" : : "r" (*(a11 + (15  )*rs_a + (11  )*cs_a)));
        __asm__ volatile ("vmcs vs14,  %0" : : "r" (*(a11 + (15  )*rs_a + (12  )*cs_a)));
        __asm__ volatile ("vmcs vs15,  %0" : : "r" (*(a11 + (15  )*rs_a + (13  )*cs_a)));
        __asm__ volatile ("vmcs vs16,  %0" : : "r" (*(a11 + (15  )*rs_a + (14  )*cs_a)));
        vf(&bli_sgemmtrsm_l_hwacha_16xn_vf_inner_15);

}
