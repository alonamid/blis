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


extern void bli_sgemm_hwacha_16xn_vf_init(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_init_beta(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_tail(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_inner_0(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_inner_1(void) __attribute__((visibility("protected")));
extern void bli_sgemm_hwacha_16xn_vf_end(void) __attribute__((visibility("protected")));

#define vf(p) \
        __asm__ __volatile__ ("vf (%0)" : : "r" (p))

/* The Hwacha vector register file can hold 4096 FP32 elements.
 * Splitting the register file between A, B, and C give 1024 elements each
 * We want C to be square (theory of matrix multiplication), so MR=NR=32
 */

void bli_sgemm_hwacha_16xn
     (
       dim_t               k0,
       float*    restrict alpha,
       float*    restrict a,
       float*    restrict b,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

	const num_t        dt     = BLIS_FLOAT;

	const dim_t        mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t        nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );

	const inc_t        packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx );
	const inc_t        packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );

	const inc_t        cs_a   = packmr;
	const inc_t        rs_b   = packnr;

       if (k0 == 0) 
       {
  
         //bli_obj_create
         //(
         //   num_t  dt,
         //   dim_t  mr,
         //   dim_t  nr,
         //   inc_t  rs_c0,
         //   inc_t  cs_c0,
         //   obj_t* obj
         //); 
         //bli_scalm( beta, c );
         bli_sscalm(
            BLIS_NO_CONJUGATE,
            0,
            BLIS_NONUNIT_DIAG,
            BLIS_DENSE,
            mr,
            nr,
            beta,
            c, rs_c0, cs_c0
         );

       } else {
/*
        printf("Starting Hwacha matmul\n");
        printf("mr: %d, nr: %d, k0: %d\n", mr, nr, k0);
        printf("a: %p, b: %p, c: %p\n", a, b, c);
        printf("cs_a: %d, rs_b: %d, cs_c0: %d, rs_c0: %d\n", cs_a, rs_b, cs_c0, rs_c0);
        printf("alpha: %f, beta: %f\n", *alpha, *beta);
        printf("alpha-hex: %x, beta-hex: %x\n", (uint32_t)*alpha, (uint32_t)*beta);
        printf("alpha-addr: %p, beta-addr: %x\n", alpha, beta);
*/
/*
        printf("===GEMM Microkernel A panel====\n");
        for (int i=0; i<mr; i++) {
           for (int k=0; k<k0; k++) {
             printf("%f ", *(a + k*cs_a + i));
          }
          printf("\n");
        }
        printf("===GEMM Microkernel B panel====\n");
        for (int i=0; i<nr; i++) {
           for (int k=0; k<k0; k++) {
             printf("%f ", *(b + k*rs_b + i));
          }
          printf("\n");
        }

        printf("===GEMM Microkernel C Bias====\n");
        for (int i=0; i<mr; i++) {
         for (int j=0; j<nr; j++) {
           printf("%f ", *(c + i*rs_c0 + j*cs_c0));
         }
          printf("\n");
        }
*/
/*
        printf("===GEMM Microkernel Manual Result C====\n");
        for (int i=0; i<mr; i++) {
         for (int j=0; j<nr; j++) {
           float cval = *beta * *(c + i*rs_c0 + j*cs_c0);
           for (int k=0; k<k0; k++) {
             cval += *alpha * *(a + k*cs_a + i) * *(b + k*rs_b + j);
           }
           printf("%f ", cval);
          }
          printf("\n");
        }
*/

          //TODO: this should be in the blis context initialization
          __asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, mr+2, 0, 1)));
      
          int vlen_result;
          __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (nr));
          if (vlen_result < nr)
          {
            printf("ERROR: vlen=%d is smaller than NR=%ld\n", vlen_result, nr);
            exit(-1);
          }
   
          float* a_ptr = a;
          float* b_ptr = b;

          // C rows addresses
          __asm__ volatile ("vmca va0,  %0" : : "r" (c+0*rs_c0));
          __asm__ volatile ("vmca va1,  %0" : : "r" (c+1*rs_c0));
          __asm__ volatile ("vmca va2,  %0" : : "r" (c+2*rs_c0));
          __asm__ volatile ("vmca va3,  %0" : : "r" (c+3*rs_c0));
          __asm__ volatile ("vmca va4,  %0" : : "r" (c+4*rs_c0));
          __asm__ volatile ("vmca va5,  %0" : : "r" (c+5*rs_c0));
          __asm__ volatile ("vmca va6,  %0" : : "r" (c+6*rs_c0));
          __asm__ volatile ("vmca va7,  %0" : : "r" (c+7*rs_c0));
          __asm__ volatile ("vmca va8,  %0" : : "r" (c+8*rs_c0));
          __asm__ volatile ("vmca va9,  %0" : : "r" (c+9*rs_c0));
          __asm__ volatile ("vmca va10, %0" : : "r" (c+10*rs_c0));
          __asm__ volatile ("vmca va11, %0" : : "r" (c+11*rs_c0));
          __asm__ volatile ("vmca va12, %0" : : "r" (c+12*rs_c0));
          __asm__ volatile ("vmca va13, %0" : : "r" (c+13*rs_c0));
          __asm__ volatile ("vmca va14, %0" : : "r" (c+14*rs_c0));
          __asm__ volatile ("vmca va15, %0" : : "r" (c+15*rs_c0));

         //B row address
         __asm__ volatile ("vmca va16, %0 \n\t" : : "r" (b_ptr));
         b_ptr += rs_b;

         // load C and first B
	 if (*beta) {
                // load beta
                __asm__ volatile ("vmcs vs63,  %0" : : "r" (*beta));
		vf(&bli_sgemm_hwacha_16xn_vf_init_beta);
	 } else {
		vf(&bli_sgemm_hwacha_16xn_vf_init);
	 }
    

         // load alpha
          __asm__ volatile ("vmcs vs63,  %0" : : "r" (*alpha));

          int k = k0; 
          for (k = k0; k > 1; k-=2)
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


          vf(&bli_sgemm_hwacha_16xn_vf_end);
	  __asm__ volatile ("fence" ::: "memory");

/*
        printf("===GEMM Microkernel Result C====\n");
        for (int i=0; i<mr; i++) {
         for (int j=0; j<nr; j++) {
           printf("%f ", *(c + i*rs_c0 + j*cs_c0));
         }
          printf("\n");
        }
*/
    }

}
