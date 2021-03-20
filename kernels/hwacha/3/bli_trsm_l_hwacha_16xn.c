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

extern void bli_strsm_hwacha_16xn_vf_inner_0(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_1(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_2(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_3(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_4(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_5(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_6(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_7(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_8(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_9(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_10(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_11(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_12(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_13(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_14(void) __attribute__((visibility("protected")));
extern void bli_strsm_hwacha_16xn_vf_inner_15(void) __attribute__((visibility("protected")));


void bli_strsm_l_hwacha_16xn
     (
       float*  restrict a11,
       float*  restrict b11,
       float*  restrict c11, inc_t rs_c, inc_t cs_c,
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

/*
        printf("===TRSM Microkernel A panel====\n");
        for (int i=0; i<m; i++) {
           for (int k=0; k<n; k++) {
             float a_f;
             bli_tofloat(*((elem_t*)a11 + k*cs_a + i), a_f);
             printf("%f ", a_f);
          }
          printf("\n");
        }
        printf("===TRSM Microkernel B panel====\n");
        for (int k=0; k<m; k++) {
          for (int i=0; i<n; i++) {
             float b_f;
             bli_tofloat(*((elem_t*)b11 + k*rs_b + i), b_f);
             printf("%f ", b_f);
          }
          printf("\n");
        }
*/

        //TODO: this should be in the blis context initialization
        __asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, mr+1, 0, 1)));

        int vlen_result;
        __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (nr));
        if (vlen_result < nr)
        {
          printf("ERROR: vlen=%d is smaller than NR=%ld\n", vlen_result, nr);
          exit(-1);
        }


        /* vv0-15 are B rows */
        // B rows addresses
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

        /* vv16-31 are C rows */
        // C rows addresses
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
        vf(&bli_strsm_hwacha_16xn_vf_inner_0);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (1  )*rs_a + (1  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (1  )*rs_a + (0  )*cs_a)));
        vf(&bli_strsm_hwacha_16xn_vf_inner_1);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (2  )*rs_a + (2  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (2  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (2  )*rs_a + (1  )*cs_a)));
        vf(&bli_strsm_hwacha_16xn_vf_inner_2);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (3  )*rs_a + (3  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (3  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (3  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (3  )*rs_a + (2  )*cs_a)));
        vf(&bli_strsm_hwacha_16xn_vf_inner_3);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (4  )*rs_a + (4  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (4  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (4  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (4  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 + (4  )*rs_a + (3  )*cs_a)));
        vf(&bli_strsm_hwacha_16xn_vf_inner_4);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (5  )*rs_a + (5  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (5  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (5  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (5  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 + (5  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 + (5  )*rs_a + (4  )*cs_a)));
        vf(&bli_strsm_hwacha_16xn_vf_inner_5);

        /* alpha_ii */
        __asm__ volatile ("vmcs vs1,  %0" : : "r" (*(a11 + (6  )*rs_a + (6  )*cs_a)));
        /* alpha_ik */
        __asm__ volatile ("vmcs vs2,  %0" : : "r" (*(a11 + (6  )*rs_a + (0  )*cs_a)));
        __asm__ volatile ("vmcs vs3,  %0" : : "r" (*(a11 + (6  )*rs_a + (1  )*cs_a)));
        __asm__ volatile ("vmcs vs4,  %0" : : "r" (*(a11 + (6  )*rs_a + (2  )*cs_a)));
        __asm__ volatile ("vmcs vs5,  %0" : : "r" (*(a11 + (6  )*rs_a + (3  )*cs_a)));
        __asm__ volatile ("vmcs vs6,  %0" : : "r" (*(a11 + (6  )*rs_a + (4  )*cs_a)));
        __asm__ volatile ("vmcs vs7,  %0" : : "r" (*(a11 + (6  )*rs_a + (5  )*cs_a)));
        vf(&bli_strsm_hwacha_16xn_vf_inner_6);

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
        vf(&bli_strsm_hwacha_16xn_vf_inner_7);

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
        vf(&bli_strsm_hwacha_16xn_vf_inner_8);

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
        vf(&bli_strsm_hwacha_16xn_vf_inner_9);
       
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
        vf(&bli_strsm_hwacha_16xn_vf_inner_10);

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
        vf(&bli_strsm_hwacha_16xn_vf_inner_11);

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
        vf(&bli_strsm_hwacha_16xn_vf_inner_12);

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
        vf(&bli_strsm_hwacha_16xn_vf_inner_13);

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
        vf(&bli_strsm_hwacha_16xn_vf_inner_14);

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
        vf(&bli_strsm_hwacha_16xn_vf_inner_15);

/*
        printf("===TRSM Microkernel Result C====\n");
        for (int ii=0; ii<mr; ii++) {
         for (int jj=0; jj<nr; jj++) {
           printf("%f ", *(c11 + ii*rs_c + jj*cs_c));
         }
          printf("\n");
        }
*/

}

