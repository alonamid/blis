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

/* The Hwacha vector register file can hold 4096 FP32 elements.
 * Splitting the register file between A, B, and C give 1024 elements each
 * We want C to be square (theory of matrix multiplication), so MR=NR=32
 */

void bli_sgemm_hwacha_16x16_unrolled
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
          __asm__ volatile ("vsetcfg %0" : : "r" (VCFG(0, 2*mr, 0, 1)));
      
          int vlen_result;
          __asm__ volatile ("vsetvl %0, %1" : "=r" (vlen_result) : "r" (mr));
          if (vlen_result < mr)
          {
            printf("ERROR: vlen=%d is smaller than MR=%ld\n", vlen_result, mr);
            exit(-1);
          }
      
          void * vpset_vfblockaddr;
          __asm__ volatile ("la %0, hwacha_sgemm_internal_vpset" : "=r" (vpset_vfblockaddr));
          __asm__ volatile ("vf 0(%0)" : : "r" (vpset_vfblockaddr));
      
          void * pre_vfblockaddr;
          void * main1_vfblockaddr;
          void * main2_vfblockaddr;
          void * main3_vfblockaddr;
          void * main4_vfblockaddr;
          void * main5_vfblockaddr;
          void * main6_vfblockaddr;
          void * tail_vfblockaddr;
          void * post_vfblockaddr;
   
          float* a_ptr = a;
          float* b_ptr = b;
 
          __asm__ volatile ("la %0, hwacha_sgemm_internal_pre" : "=r" (pre_vfblockaddr));
          __asm__ volatile ("la %0, hwacha_sgemm_internal_main_p1" : "=r" (main1_vfblockaddr));
          __asm__ volatile ("la %0, hwacha_sgemm_internal_main_p2" : "=r" (main2_vfblockaddr));
          __asm__ volatile ("la %0, hwacha_sgemm_internal_main_p3" : "=r" (main3_vfblockaddr));
          __asm__ volatile ("la %0, hwacha_sgemm_internal_main_p4" : "=r" (main4_vfblockaddr));
          __asm__ volatile ("la %0, hwacha_sgemm_internal_main_p5" : "=r" (main5_vfblockaddr));
          __asm__ volatile ("la %0, hwacha_sgemm_internal_main_p6" : "=r" (main6_vfblockaddr));
          __asm__ volatile ("la %0, hwacha_sgemm_internal_post" : "=r" (post_vfblockaddr));

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

         // load beta
          __asm__ volatile ("vmcs vs61,  %0" : : "r" (*beta));
     
          //load C           
          __asm__ volatile ("vf 0(%0)" : : "r" (pre_vfblockaddr));

         // load alpha
          __asm__ volatile ("vmcs vs62,  %0" : : "r" (*alpha));

          int k = k0; 
          for (; k > 15; k-=16) {
		printf("inside main loop\n");
     
              // B rows 1-16
              __asm__ volatile ("vmca va16, %0 \n\t"
                            "vmca va17, %1 \n\t"
                            "vmca va18, %2 \n\t"
                            "vmca va19, %3 \n\t"
      
                            "vmca va20, %4 \n\t"
                            "vmca va21, %5 \n\t"
                            "vmca va22, %6 \n\t"
                            "vmca va23, %7 \n\t"
      
                            "vmca va24, %8 \n\t"
                            "vmca va25, %9 \n\t"
                            "vmca va26, %10\n\t"
                            "vmca va27, %11\n\t"
      
                            "vmca va28, %12\n\t"
                            "vmca va29, %13\n\t"
                            "vmca va30, %14\n\t"
                            "vmca va31, %15\n\t"
                        : 
                        : "r" (b_ptr+0*16),  "r" (b_ptr+1*16),  "r" (b_ptr+2*16),  "r" (b_ptr+3*16), 
                          "r" (b_ptr+4*16),  "r" (b_ptr+5*16),  "r" (b_ptr+6*16),  "r" (b_ptr+7*16), 
                          "r" (b_ptr+8*16),  "r" (b_ptr+9*16),  "r" (b_ptr+10*16), "r" (b_ptr+11*16), 
                          "r" (b_ptr+12*16), "r" (b_ptr+13*16), "r" (b_ptr+14*16), "r" (b_ptr+15*16) 
                      );

              b_ptr += 16*16;

              for (int p=0; p<5; p++) {

                // A column 1
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
        
  
                a_ptr = a_ptr + 16;
  
                // A column 2
                __asm__ volatile ("vmcs vs17, %0 \n\t"
                              "vmcs vs18, %1 \n\t"
                              "vmcs vs19, %2 \n\t"
                              "vmcs vs20, %3 \n\t"
        
                              "vmcs vs21, %4 \n\t"
                              "vmcs vs22, %5 \n\t"
                              "vmcs vs23, %6 \n\t"
                              "vmcs vs24, %7 \n\t"
        
                              "vmcs vs25, %8 \n\t"
                              "vmcs vs26, %9 \n\t"
                              "vmcs vs27, %10\n\t"
                              "vmcs vs28, %11\n\t"
        
                              "vmcs vs29, %12\n\t"
                              "vmcs vs30, %13\n\t"
                              "vmcs vs31, %14\n\t"
                              "vmcs vs32, %15\n\t"
                          : 
                          : "r" (*(a_ptr+0)),  "r" (*(a_ptr+1)),  "r" (*(a_ptr+2)),  "r" (*(a_ptr+3)), 
                            "r" (*(a_ptr+4)),  "r" (*(a_ptr+5)),  "r" (*(a_ptr+6)),  "r" (*(a_ptr+7)), 
                            "r" (*(a_ptr+8)),  "r" (*(a_ptr+9)),  "r" (*(a_ptr+10)), "r" (*(a_ptr+11)), 
                            "r" (*(a_ptr+12)), "r" (*(a_ptr+13)), "r" (*(a_ptr+14)), "r" (*(a_ptr+15)) 
                        );
        
  
                a_ptr = a_ptr + 16;
  
                // A column 3
                __asm__ volatile ("vmcs vs33, %0 \n\t"
                              "vmcs vs34, %1 \n\t"
                              "vmcs vs35, %2 \n\t"
                              "vmcs vs36, %3 \n\t"
        
                              "vmcs vs37, %4 \n\t"
                              "vmcs vs38, %5 \n\t"
                              "vmcs vs39, %6 \n\t"
                              "vmcs vs40, %7 \n\t"
        
                              "vmcs vs41, %8 \n\t"
                              "vmcs vs42, %9 \n\t"
                              "vmcs vs43, %10\n\t"
                              "vmcs vs44, %11\n\t"
        
                              "vmcs vs45, %12\n\t"
                              "vmcs vs46, %13\n\t"
                              "vmcs vs47, %14\n\t"
                              "vmcs vs48, %15\n\t"
                          : 
                          : "r" (*(a_ptr+0)),  "r" (*(a_ptr+1)),  "r" (*(a_ptr+2)),  "r" (*(a_ptr+3)), 
                            "r" (*(a_ptr+4)),  "r" (*(a_ptr+5)),  "r" (*(a_ptr+6)),  "r" (*(a_ptr+7)), 
                            "r" (*(a_ptr+8)),  "r" (*(a_ptr+9)),  "r" (*(a_ptr+10)), "r" (*(a_ptr+11)), 
                            "r" (*(a_ptr+12)), "r" (*(a_ptr+13)), "r" (*(a_ptr+14)), "r" (*(a_ptr+15)) 
                        );
        
  
                a_ptr = a_ptr + 16;

                switch(p) {
                   case 0  :
                      __asm__ volatile ("vf 0(%0)" : : "r" (main1_vfblockaddr));
                      break;	
                   case 1  :
                      __asm__ volatile ("vf 0(%0)" : : "r" (main2_vfblockaddr));
                      break;
                   case 2  :
                      __asm__ volatile ("vf 0(%0)" : : "r" (main3_vfblockaddr));
                      break;
                   case 3  :
                      __asm__ volatile ("vf 0(%0)" : : "r" (main4_vfblockaddr));
                      break;
                   case 4  :
                      __asm__ volatile ("vf 0(%0)" : : "r" (main5_vfblockaddr));
                      break;
                }  

              }

              // A column 16
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
 
              a_ptr += 16;
              
              __asm__ volatile ("vf 0(%0)" : : "r" (main6_vfblockaddr));
          }

          while (k > 0)
          {
		printf("inside tail loop\n");
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
              __asm__ volatile ("vmca va16, %0 \n\t" : : "r" (b_ptr+0*16));

              //outer-product 
              __asm__ volatile ("la %0, hwacha_sgemm_internal_tail" : "=r" (tail_vfblockaddr));
              __asm__ volatile ("vf 0(%0)" : : "r" (tail_vfblockaddr));

              a_ptr += 16;
              b_ptr += 16;
            
              k--; 
          }



          __asm__ volatile ("vf 0(%0)" : : "r" (post_vfblockaddr));

          //jump after the various vf code blocks
          __asm__ volatile("j end_vf \n\t" ::);
    
          __asm__ volatile(".align 3                     \n\t"
                       "hwacha_sgemm_internal_vpset: \n\t"
                       "vpset vp0                    \n\t"
                       "vstop                        \n\t" : :);
   
          //load C and multiply by beta, vector fetch block 
          __asm__ volatile(".align 3                       \n\t"
                       "hwacha_sgemm_internal_pre:     \n\t"
                       "vlw vv0,  va0                  \n\t"
                       "vfmul.s vv0, vv0, vs61          \n\t" 
                       "vlw vv1,  va1                  \n\t"
                       "vfmul.s vv1, vv1, vs61          \n\t" 
                       "vlw vv2,  va2                  \n\t"
                       "vfmul.s vv2, vv2, vs61          \n\t" 
                       "vlw vv3,  va3                  \n\t"
                       "vfmul.s vv3, vv3, vs61          \n\t" 
                       "vlw vv4,  va4                  \n\t"
                       "vfmul.s vv4, vv4, vs61          \n\t" 
                       "vlw vv5,  va5                  \n\t"
                       "vfmul.s vv5, vv5, vs61          \n\t" 
                       "vlw vv6,  va6                  \n\t"
                       "vfmul.s vv6, vv6, vs61          \n\t" 
                       "vlw vv7,  va7                  \n\t"
                       "vfmul.s vv7, vv7, vs61          \n\t" 
                       "vlw vv8,  va8                  \n\t"
                       "vfmul.s vv8, vv8, vs61          \n\t" 
                       "vlw vv9,  va9                  \n\t"
                       "vfmul.s vv9, vv9, vs61          \n\t" 
                       "vlw vv10, va10                 \n\t"
                       "vfmul.s vv10, vv10, vs61        \n\t" 
                       "vlw vv11, va11                 \n\t"
                       "vfmul.s vv11, vv11, vs61        \n\t" 
                       "vlw vv12, va12                 \n\t"
                       "vfmul.s vv12, vv12, vs61        \n\t" 
                       "vlw vv13, va13                 \n\t"
                       "vfmul.s vv13, vv13, vs61        \n\t" 
                       "vlw vv14, va14                 \n\t"
                       "vfmul.s vv14, vv14, vs61        \n\t" 
                       "vlw vv15, va15                 \n\t"
                       "vfmul.s vv15, vv15, vs61        \n\t" 
                       "vstop                          \n\t" : :);

          //load A, B and matmul vector fetch block
          __asm__ volatile(".align 3                       \n\t"
                       "hwacha_sgemm_internal_main_p1: \n\t"
                       "vlw vv16, va16                 \n\t"   /* # b0             */
                       "vfmul.s vv16, vv16, vs62        \n\t"   /* # b0 * alpha     */
                       "vfmadd.s vv0, vv16, vs1, vv0   \n\t"   /* # c0 += a00 * b0 */
                       "vfmadd.s vv1, vv16, vs2, vv1   \n\t"   /* # c1 += a10 * b0 */
                    
                       "vlw vv17, va17                 \n\t"   /* # b1             */
                       "vfmul.s vv17, vv17, vs62        \n\t"   /* # b1 * alpha     */
                       "vfmadd.s vv2, vv16, vs3, vv2   \n\t"   /* # c2 += a20 * b0 */
                       "vfmadd.s vv3, vv16, vs4, vv3   \n\t"   /* # c3 += a30 * b0 */
                    
                       "vlw vv18, va18                 \n\t"   /* # b2             */
                       "vfmul.s vv18, vv18, vs62        \n\t"   /* # b2 * alpha     */
                       "vfmadd.s vv4, vv16, vs5, vv4   \n\t"   /* # c4 += a40 * b0 */
                       "vfmadd.s vv5, vv16, vs6, vv5   \n\t"   /* # c5 += a50 * b0 */
                    
                       "vlw vv19, va19                 \n\t"   /* # b3             */
                       "vfmul.s vv19, vv19, vs62        \n\t"   /* # b3 * alpha     */
                       "vfmadd.s vv6, vv16, vs7, vv6   \n\t"   /* # c6 += a60 * b0 */
                       "vfmadd.s vv7, vv16, vs8, vv7   \n\t"   /* # c7 += a70 * b0 */
 
                       "vlw vv20, va20                 \n\t"   /* # b4             */
                       "vfmul.s vv20, vv20, vs62        \n\t"   /* # b4 * alpha     */
                       "vfmadd.s vv8, vv16, vs9, vv8   \n\t"   /* # c8 += a80 * b0 */
                       "vfmadd.s vv9, vv16, vs10, vv9  \n\t"   /* # c9 += a90 * b0 */

                       "vlw vv21, va21                 \n\t"   /* # b5               */
                       "vfmul.s vv21, vv21, vs62        \n\t"   /* # b5 * alpha     */
                       "vfmadd.s vv10, vv16, vs11, vv10\n\t"   /* # c10 += a100 * b0 */
                       "vfmadd.s vv11, vv16, vs12, vv11\n\t"   /* # c11 += a110 * b0 */

                       "vlw vv22, va22                 \n\t"   /* # b6               */
                       "vfmul.s vv22, vv22, vs62        \n\t"   /* # b6 * alpha     */
                       "vfmadd.s vv12, vv16, vs13, vv12\n\t"   /* # c12 += a120 * b0 */
                       "vfmadd.s vv13, vv16, vs14, vv13\n\t"   /* # c13 += a130 * b0 */

                       "vlw vv23, va23                 \n\t"   /* # b7               */
                       "vfmul.s vv23, vv23, vs62        \n\t"   /* # b7 * alpha     */
                       "vfmadd.s vv14, vv16, vs15, vv14\n\t"   /* # c14 += a140 * b0 */
                       "vfmadd.s vv15, vv16, vs16, vv15\n\t"   /* # c15 += a150 * b0 */

                       "vlw vv24, va24                 \n\t"   /* # b8               */
                       "vfmul.s vv24, vv24, vs62        \n\t"   /* # b8 * alpha     */
                       "vfmadd.s vv0, vv17, vs17, vv0  \n\t"   /* # c0 += a01 * b1   */
                       "vfmadd.s vv1, vv17, vs18, vv1  \n\t"   /* # c1 += a11 * b1   */

                       "vlw vv25, va25                 \n\t"   /* # b9               */
                       "vfmul.s vv25, vv25, vs62        \n\t"   /* # b9 * alpha     */
                       "vfmadd.s vv2, vv17, vs19, vv2  \n\t"   /* # c2 += a21 * b1   */
                       "vfmadd.s vv3, vv17, vs20, vv3  \n\t"   /* # c3 += a31 * b1   */

                       "vlw vv26, va26                 \n\t"   /* # b10              */
                       "vfmul.s vv26, vv26, vs62        \n\t"   /* # b10 * alpha     */
                       "vfmadd.s vv4, vv17, vs21, vv4  \n\t"   /* # c4 += a41 * b1   */
                       "vfmadd.s vv5, vv17, vs22, vv5  \n\t"   /* # c5 += a51 * b1   */

                       "vlw vv27, va27                 \n\t"   /* # b11              */
                       "vfmul.s vv27, vv27, vs62        \n\t"   /* # b11 * alpha     */
                       "vfmadd.s vv6, vv17, vs23, vv6  \n\t"   /* # c6 += a61 * b1   */
                       "vfmadd.s vv7, vv17, vs24, vv7  \n\t"   /* # c7 += a71 * b1   */

                       "vlw vv28, va28                 \n\t"   /* # b12              */
                       "vfmul.s vv28, vv28, vs62        \n\t"   /* # b12 * alpha     */
                       "vfmadd.s vv8, vv17, vs25, vv8  \n\t"   /* # c8 += a81 * b1   */
                       "vfmadd.s vv9, vv17, vs26, vv9  \n\t"   /* # c9 += a91 * b1   */

                       "vlw vv29, va29                 \n\t"   /* # b13              */
                       "vfmul.s vv29, vv29, vs62        \n\t"   /* # b13 * alpha     */
                       "vfmadd.s vv10, vv17, vs27, vv10\n\t"   /* # c10 += a101 * b1 */
                       "vfmadd.s vv11, vv17, vs28, vv11\n\t"   /* # c11 += a111 * b1 */

                       "vlw vv30, va30                 \n\t"   /* # b14              */
                       "vfmul.s vv30, vv30, vs62        \n\t"   /* # b14 * alpha     */
                       "vfmadd.s vv12, vv17, vs29, vv12\n\t"   /* # c12 += a121 * b1 */
                       "vfmadd.s vv13, vv17, vs30, vv13\n\t"   /* # c13 += a131 * b1 */

                       "vlw vv31, va31                 \n\t"   /* # b15              */
                       "vfmul.s vv31, vv31, vs62        \n\t"   /* # b15 * alpha     */
                       "vfmadd.s vv14, vv17, vs31, vv14\n\t"   /* # c14 += a141 * b1 */
                       "vfmadd.s vv15, vv17, vs32, vv15\n\t"   /* # c15 += a151 * b1 */

                       "vfmadd.s vv0,  vv18, vs33, vv0 \n\t"   /* # c0 +=  a02  * b2 */
                       "vfmadd.s vv1,  vv18, vs34, vv1 \n\t"   /* # c1 +=  a12  * b2 */
                       "vfmadd.s vv2,  vv18, vs35, vv2 \n\t"   /* # c2 +=  a22  * b2 */
                       "vfmadd.s vv3,  vv18, vs36, vv3 \n\t"   /* # c3 +=  a32  * b2 */
                       "vfmadd.s vv4,  vv18, vs37, vv4 \n\t"   /* # c4 +=  a42  * b2 */
                       "vfmadd.s vv5,  vv18, vs38, vv5 \n\t"   /* # c5 +=  a52  * b2 */
                       "vfmadd.s vv6,  vv18, vs39, vv6 \n\t"   /* # c6 +=  a62  * b2 */
                       "vfmadd.s vv7,  vv18, vs30, vv7 \n\t"   /* # c7 +=  a72  * b2 */
                       "vfmadd.s vv8,  vv18, vs41, vv8 \n\t"   /* # c8 +=  a82  * b2 */
                       "vfmadd.s vv9,  vv18, vs42, vv9 \n\t"   /* # c9 +=  a92  * b2 */
                       "vfmadd.s vv10, vv18, vs43, vv10\n\t"   /* # c10 += a102 * b2 */
                       "vfmadd.s vv11, vv18, vs44, vv11\n\t"   /* # c11 += a112 * b2 */
                       "vfmadd.s vv12, vv18, vs45, vv12\n\t"   /* # c12 += a122 * b2 */
                       "vfmadd.s vv13, vv18, vs46, vv13\n\t"   /* # c13 += a132 * b2 */
                       "vfmadd.s vv14, vv18, vs47, vv14\n\t"   /* # c14 += a142 * b2 */
                       "vfmadd.s vv15, vv18, vs48, vv15\n\t"   /* # c15 += a152 * b2 */
 
                       "vstop                          \n\t" : : );


          __asm__ volatile(".align 3                       \n\t"
                       "hwacha_sgemm_internal_main_p2: \n\t"
                       "vfmadd.s vv0,  vv19, vs1, vv0  \n\t"   /* # c0 +=  a03  * b3 */
                       "vfmadd.s vv1,  vv19, vs2, vv1  \n\t"   /* # c1 +=  a13  * b3 */
                       "vfmadd.s vv2,  vv19, vs3, vv2  \n\t"   /* # c2 +=  a23  * b3 */
                       "vfmadd.s vv3,  vv19, vs4, vv3  \n\t"   /* # c3 +=  a33  * b3 */
                       "vfmadd.s vv4,  vv19, vs5, vv4  \n\t"   /* # c4 +=  a43  * b3 */
                       "vfmadd.s vv5,  vv19, vs6, vv5  \n\t"   /* # c5 +=  a53  * b3 */
                       "vfmadd.s vv6,  vv19, vs7, vv6  \n\t"   /* # c6 +=  a63  * b3 */
                       "vfmadd.s vv7,  vv19, vs8, vv7  \n\t"   /* # c7 +=  a73  * b3 */
                       "vfmadd.s vv8,  vv19, vs9, vv8  \n\t"   /* # c8 +=  a83  * b3 */
                       "vfmadd.s vv9,  vv19, vs10, vv9 \n\t"   /* # c9 +=  a93  * b3 */
                       "vfmadd.s vv10, vv19, vs11, vv10\n\t"   /* # c10 += a103 * b3 */
                       "vfmadd.s vv11, vv19, vs12, vv11\n\t"   /* # c11 += a113 * b3 */
                       "vfmadd.s vv12, vv19, vs13, vv12\n\t"   /* # c12 += a123 * b3 */
                       "vfmadd.s vv13, vv19, vs14, vv13\n\t"   /* # c13 += a133 * b3 */
                       "vfmadd.s vv14, vv19, vs15, vv14\n\t"   /* # c14 += a143 * b3 */
                       "vfmadd.s vv15, vv19, vs16, vv15\n\t"   /* # c15 += a153 * b3 */
                       "vfmadd.s vv0,  vv20, vs17, vv0 \n\t"   /* # c0 +=  a04  * b4 */
                       "vfmadd.s vv1,  vv20, vs18, vv1 \n\t"   /* # c1 +=  a14  * b4 */
                       "vfmadd.s vv2,  vv20, vs19, vv2 \n\t"   /* # c2 +=  a24  * b4 */
                       "vfmadd.s vv3,  vv20, vs20, vv3 \n\t"   /* # c3 +=  a34  * b4 */
                       "vfmadd.s vv4,  vv20, vs21, vv4 \n\t"   /* # c4 +=  a44  * b4 */
                       "vfmadd.s vv5,  vv20, vs22, vv5 \n\t"   /* # c5 +=  a54  * b4 */
                       "vfmadd.s vv6,  vv20, vs23, vv6 \n\t"   /* # c6 +=  a64  * b4 */
                       "vfmadd.s vv7,  vv20, vs24, vv7 \n\t"   /* # c7 +=  a74  * b4 */
                       "vfmadd.s vv8,  vv20, vs25, vv8 \n\t"   /* # c8 +=  a84  * b4 */
                       "vfmadd.s vv9,  vv20, vs26, vv9 \n\t"   /* # c9 +=  a94  * b4 */
                       "vfmadd.s vv10, vv20, vs27, vv10\n\t"   /* # c10 += a104 * b4 */
                       "vfmadd.s vv11, vv20, vs28, vv11\n\t"   /* # c11 += a114 * b4 */
                       "vfmadd.s vv12, vv20, vs29, vv12\n\t"   /* # c12 += a124 * b4 */
                       "vfmadd.s vv13, vv20, vs30, vv13\n\t"   /* # c13 += a134 * b4 */
                       "vfmadd.s vv14, vv20, vs31, vv14\n\t"   /* # c14 += a144 * b4 */
                       "vfmadd.s vv15, vv20, vs32, vv15\n\t"   /* # c15 += a154 * b4 */
                       "vfmadd.s vv0,  vv21, vs33, vv0 \n\t"   /* # c0 +=  a05  * b5 */
                       "vfmadd.s vv1,  vv21, vs34, vv1 \n\t"   /* # c1 +=  a15  * b5 */
                       "vfmadd.s vv2,  vv21, vs35, vv2 \n\t"   /* # c2 +=  a25  * b5 */
                       "vfmadd.s vv3,  vv21, vs36, vv3 \n\t"   /* # c3 +=  a35  * b5 */
                       "vfmadd.s vv4,  vv21, vs37, vv4 \n\t"   /* # c4 +=  a45  * b5 */
                       "vfmadd.s vv5,  vv21, vs38, vv5 \n\t"   /* # c5 +=  a55  * b5 */
                       "vfmadd.s vv6,  vv21, vs39, vv6 \n\t"   /* # c6 +=  a65  * b5 */
                       "vfmadd.s vv7,  vv21, vs40, vv7 \n\t"   /* # c7 +=  a75  * b5 */
                       "vfmadd.s vv8,  vv21, vs41, vv8 \n\t"   /* # c8 +=  a85  * b5 */
                       "vfmadd.s vv9,  vv21, vs42, vv9 \n\t"   /* # c9 +=  a95  * b5 */
                       "vfmadd.s vv10, vv21, vs43, vv10\n\t"   /* # c10 += a105 * b5 */
                       "vfmadd.s vv11, vv21, vs44, vv11\n\t"   /* # c11 += a115 * b5 */
                       "vfmadd.s vv12, vv21, vs45, vv12\n\t"   /* # c12 += a125 * b5 */
                       "vfmadd.s vv13, vv21, vs46, vv13\n\t"   /* # c13 += a135 * b5 */
                       "vfmadd.s vv14, vv21, vs47, vv14\n\t"   /* # c14 += a145 * b5 */
                       "vfmadd.s vv15, vv21, vs48, vv15\n\t"   /* # c15 += a155 * b5 */
 
                       "vstop                          \n\t" : : );

          __asm__ volatile(".align 3                       \n\t"
                       "hwacha_sgemm_internal_main_p3: \n\t"
                       "vfmadd.s vv0,  vv22, vs1, vv0  \n\t"   /* # c0 +=  a06  * b6 */
                       "vfmadd.s vv1,  vv22, vs2, vv1  \n\t"   /* # c1 +=  a16  * b6 */
                       "vfmadd.s vv2,  vv22, vs3, vv2  \n\t"   /* # c2 +=  a26  * b6 */
                       "vfmadd.s vv3,  vv22, vs4, vv3  \n\t"   /* # c3 +=  a36  * b6 */
                       "vfmadd.s vv4,  vv22, vs5, vv4  \n\t"   /* # c4 +=  a46  * b6 */
                       "vfmadd.s vv5,  vv22, vs6, vv5  \n\t"   /* # c5 +=  a56  * b6 */
                       "vfmadd.s vv6,  vv22, vs7, vv6  \n\t"   /* # c6 +=  a66  * b6 */
                       "vfmadd.s vv7,  vv22, vs8, vv7  \n\t"   /* # c7 +=  a76  * b6 */
                       "vfmadd.s vv8,  vv22, vs9, vv8  \n\t"   /* # c8 +=  a86  * b6 */
                       "vfmadd.s vv9,  vv22, vs10, vv9 \n\t"   /* # c9 +=  a96  * b6 */
                       "vfmadd.s vv10, vv22, vs11, vv10\n\t"   /* # c10 += a106 * b6 */
                       "vfmadd.s vv11, vv22, vs12, vv11\n\t"   /* # c11 += a116 * b6 */
                       "vfmadd.s vv12, vv22, vs13, vv12\n\t"   /* # c12 += a126 * b6 */
                       "vfmadd.s vv13, vv22, vs14, vv13\n\t"   /* # c13 += a136 * b6 */
                       "vfmadd.s vv14, vv22, vs15, vv14\n\t"   /* # c14 += a146 * b6 */
                       "vfmadd.s vv15, vv22, vs16, vv15\n\t"   /* # c15 += a156 * b6 */
                       "vfmadd.s vv0,  vv23, vs17, vv0 \n\t"   /* # c0 +=  a07  * b7 */
                       "vfmadd.s vv1,  vv23, vs18, vv1 \n\t"   /* # c1 +=  a17  * b7 */
                       "vfmadd.s vv2,  vv23, vs19, vv2 \n\t"   /* # c2 +=  a27  * b7 */
                       "vfmadd.s vv3,  vv23, vs20, vv3 \n\t"   /* # c3 +=  a37  * b7 */
                       "vfmadd.s vv4,  vv23, vs21, vv4 \n\t"   /* # c4 +=  a47  * b7 */
                       "vfmadd.s vv5,  vv23, vs22, vv5 \n\t"   /* # c5 +=  a57  * b7 */
                       "vfmadd.s vv6,  vv23, vs23, vv6 \n\t"   /* # c6 +=  a67  * b7 */
                       "vfmadd.s vv7,  vv23, vs24, vv7 \n\t"   /* # c7 +=  a77  * b7 */
                       "vfmadd.s vv8,  vv23, vs25, vv8 \n\t"   /* # c8 +=  a87  * b7 */
                       "vfmadd.s vv9,  vv23, vs26, vv9 \n\t"   /* # c9 +=  a97  * b7 */
                       "vfmadd.s vv10, vv23, vs27, vv10\n\t"   /* # c10 += a107 * b7 */
                       "vfmadd.s vv11, vv23, vs28, vv11\n\t"   /* # c11 += a117 * b7 */
                       "vfmadd.s vv12, vv23, vs29, vv12\n\t"   /* # c12 += a127 * b7 */
                       "vfmadd.s vv13, vv23, vs30, vv13\n\t"   /* # c13 += a137 * b7 */
                       "vfmadd.s vv14, vv23, vs31, vv14\n\t"   /* # c14 += a147 * b7 */
                       "vfmadd.s vv15, vv23, vs32, vv15\n\t"   /* # c15 += a157 * b7 */
                       "vfmadd.s vv0,  vv24, vs33, vv0 \n\t"   /* # c0 +=  a08  * b8 */
                       "vfmadd.s vv1,  vv24, vs34, vv1 \n\t"   /* # c1 +=  a18  * b8 */
                       "vfmadd.s vv2,  vv24, vs35, vv2 \n\t"   /* # c2 +=  a28  * b8 */
                       "vfmadd.s vv3,  vv24, vs36, vv3 \n\t"   /* # c3 +=  a38  * b8 */
                       "vfmadd.s vv4,  vv24, vs37, vv4 \n\t"   /* # c4 +=  a48  * b8 */
                       "vfmadd.s vv5,  vv24, vs38, vv5 \n\t"   /* # c5 +=  a58  * b8 */
                       "vfmadd.s vv6,  vv24, vs39, vv6 \n\t"   /* # c6 +=  a68  * b8 */
                       "vfmadd.s vv7,  vv24, vs40, vv7 \n\t"   /* # c7 +=  a78  * b8 */
                       "vfmadd.s vv8,  vv24, vs41, vv8 \n\t"   /* # c8 +=  a88  * b8 */
                       "vfmadd.s vv9,  vv24, vs42, vv9 \n\t"   /* # c9 +=  a98  * b8 */
                       "vfmadd.s vv10, vv24, vs43, vv10\n\t"   /* # c10 += a108 * b8 */
                       "vfmadd.s vv11, vv24, vs44, vv11\n\t"   /* # c11 += a118 * b8 */
                       "vfmadd.s vv12, vv24, vs45, vv12\n\t"   /* # c12 += a128 * b8 */
                       "vfmadd.s vv13, vv24, vs46, vv13\n\t"   /* # c13 += a138 * b8 */
                       "vfmadd.s vv14, vv24, vs47, vv14\n\t"   /* # c14 += a148 * b8 */
                       "vfmadd.s vv15, vv24, vs48, vv15\n\t"   /* # c15 += a158 * b8 */
 
                       "vstop                          \n\t" : : );

          __asm__ volatile(".align 3                       \n\t"
                       "hwacha_sgemm_internal_main_p4: \n\t"
                       "vfmadd.s vv0,  vv25, vs1, vv0  \n\t"   /* # c0 +=  a09  * b9  */
                       "vfmadd.s vv1,  vv25, vs2, vv1  \n\t"   /* # c1 +=  a19  * b9  */
                       "vfmadd.s vv2,  vv25, vs3, vv2  \n\t"   /* # c2 +=  a29  * b9  */
                       "vfmadd.s vv3,  vv25, vs4, vv3  \n\t"   /* # c3 +=  a39  * b9  */
                       "vfmadd.s vv4,  vv25, vs5, vv4  \n\t"   /* # c4 +=  a49  * b9  */
                       "vfmadd.s vv5,  vv25, vs6, vv5  \n\t"   /* # c5 +=  a59  * b9  */
                       "vfmadd.s vv6,  vv25, vs7, vv6  \n\t"   /* # c6 +=  a69  * b9  */
                       "vfmadd.s vv7,  vv25, vs8, vv7  \n\t"   /* # c7 +=  a79  * b9  */
                       "vfmadd.s vv8,  vv25, vs9, vv8  \n\t"   /* # c8 +=  a89  * b9  */
                       "vfmadd.s vv9,  vv25, vs10, vv9 \n\t"   /* # c9 +=  a99  * b9  */
                       "vfmadd.s vv10, vv25, vs11, vv10\n\t"   /* # c10 += a109 * b9  */
                       "vfmadd.s vv11, vv25, vs12, vv11\n\t"   /* # c11 += a119 * b9  */
                       "vfmadd.s vv12, vv25, vs13, vv12\n\t"   /* # c12 += a129 * b9  */
                       "vfmadd.s vv13, vv25, vs14, vv13\n\t"   /* # c13 += a139 * b9  */
                       "vfmadd.s vv14, vv25, vs15, vv14\n\t"   /* # c14 += a149 * b9  */
                       "vfmadd.s vv15, vv25, vs16, vv15\n\t"   /* # c15 += a159 * b9  */
                       "vfmadd.s vv0,  vv26, vs17, vv0 \n\t"   /* # c0 +=  a010  * b10*/
                       "vfmadd.s vv1,  vv26, vs18, vv1 \n\t"   /* # c1 +=  a110  * b10*/
                       "vfmadd.s vv2,  vv26, vs19, vv2 \n\t"   /* # c2 +=  a210  * b10*/
                       "vfmadd.s vv3,  vv26, vs20, vv3 \n\t"   /* # c3 +=  a310  * b10*/
                       "vfmadd.s vv4,  vv26, vs21, vv4 \n\t"   /* # c4 +=  a410  * b10*/
                       "vfmadd.s vv5,  vv26, vs22, vv5 \n\t"   /* # c5 +=  a510  * b10*/
                       "vfmadd.s vv6,  vv26, vs23, vv6 \n\t"   /* # c6 +=  a610  * b10*/
                       "vfmadd.s vv7,  vv26, vs24, vv7 \n\t"   /* # c7 +=  a710  * b10*/
                       "vfmadd.s vv8,  vv26, vs25, vv8 \n\t"   /* # c8 +=  a810  * b10*/
                       "vfmadd.s vv9,  vv26, vs26, vv9 \n\t"   /* # c9 +=  a910  * b10*/
                       "vfmadd.s vv10, vv26, vs27, vv10\n\t"   /* # c10 += a1010 * b10*/
                       "vfmadd.s vv11, vv26, vs28, vv11\n\t"   /* # c11 += a1110 * b10*/
                       "vfmadd.s vv12, vv26, vs29, vv12\n\t"   /* # c12 += a1210 * b10*/
                       "vfmadd.s vv13, vv26, vs30, vv13\n\t"   /* # c13 += a1310 * b10*/
                       "vfmadd.s vv14, vv26, vs31, vv14\n\t"   /* # c14 += a1410 * b10*/
                       "vfmadd.s vv15, vv26, vs32, vv15\n\t"   /* # c15 += a1510 * b10*/
                       "vfmadd.s vv0,  vv27, vs33, vv0 \n\t"   /* # c0 +=  a011  * b11*/
                       "vfmadd.s vv1,  vv27, vs34, vv1 \n\t"   /* # c1 +=  a111  * b11*/
                       "vfmadd.s vv2,  vv27, vs35, vv2 \n\t"   /* # c2 +=  a211  * b11*/
                       "vfmadd.s vv3,  vv27, vs36, vv3 \n\t"   /* # c3 +=  a311  * b11*/
                       "vfmadd.s vv4,  vv27, vs37, vv4 \n\t"   /* # c4 +=  a411  * b11*/
                       "vfmadd.s vv5,  vv27, vs38, vv5 \n\t"   /* # c5 +=  a511  * b11*/
                       "vfmadd.s vv6,  vv27, vs39, vv6 \n\t"   /* # c6 +=  a611  * b11*/
                       "vfmadd.s vv7,  vv27, vs40, vv7 \n\t"   /* # c7 +=  a711  * b11*/
                       "vfmadd.s vv8,  vv27, vs41, vv8 \n\t"   /* # c8 +=  a811  * b11*/
                       "vfmadd.s vv9,  vv27, vs42, vv9 \n\t"   /* # c9 +=  a911  * b11*/
                       "vfmadd.s vv10, vv27, vs43, vv10\n\t"   /* # c10 += a1011 * b11*/
                       "vfmadd.s vv11, vv27, vs44, vv11\n\t"   /* # c11 += a1111 * b11*/
                       "vfmadd.s vv12, vv27, vs45, vv12\n\t"   /* # c12 += a1211 * b11*/
                       "vfmadd.s vv13, vv27, vs46, vv13\n\t"   /* # c13 += a1311 * b11*/
                       "vfmadd.s vv14, vv27, vs47, vv14\n\t"   /* # c14 += a1411 * b11*/
                       "vfmadd.s vv15, vv27, vs48, vv15\n\t"   /* # c15 += a1511 * b11*/
 
                       "vstop                          \n\t" : : );

          __asm__ volatile(".align 3                       \n\t"
                       "hwacha_sgemm_internal_main_p5: \n\t"
                       "vfmadd.s vv0,  vv28, vs1, vv0  \n\t"  /* # c0 +=  a012  * b12 */
                       "vfmadd.s vv1,  vv28, vs2, vv1  \n\t"  /* # c1 +=  a112  * b12 */
                       "vfmadd.s vv2,  vv28, vs3, vv2  \n\t"  /* # c2 +=  a212  * b12 */
                       "vfmadd.s vv3,  vv28, vs4, vv3  \n\t"  /* # c3 +=  a312  * b12 */
                       "vfmadd.s vv4,  vv28, vs5, vv4  \n\t"  /* # c4 +=  a412  * b12 */
                       "vfmadd.s vv5,  vv28, vs6, vv5  \n\t"  /* # c5 +=  a512  * b12 */
                       "vfmadd.s vv6,  vv28, vs7, vv6  \n\t"  /* # c6 +=  a612  * b12 */
                       "vfmadd.s vv7,  vv28, vs8, vv7  \n\t"  /* # c7 +=  a712  * b12 */
                       "vfmadd.s vv8,  vv28, vs9, vv8  \n\t"  /* # c8 +=  a812  * b12 */
                       "vfmadd.s vv9,  vv28, vs10, vv9 \n\t"  /* # c9 +=  a912  * b12 */
                       "vfmadd.s vv10, vv28, vs11, vv10\n\t"  /* # c10 += a1012 * b12 */
                       "vfmadd.s vv11, vv28, vs12, vv11\n\t"  /* # c11 += a1112 * b12 */
                       "vfmadd.s vv12, vv28, vs13, vv12\n\t"  /* # c12 += a1212 * b12 */
                       "vfmadd.s vv13, vv28, vs14, vv13\n\t"  /* # c13 += a1312 * b12 */
                       "vfmadd.s vv14, vv28, vs15, vv14\n\t"  /* # c14 += a1412 * b12 */
                       "vfmadd.s vv15, vv28, vs16, vv15\n\t"  /* # c15 += a1512 * b12 */
                       "vfmadd.s vv0,  vv29, vs17, vv0 \n\t"  /* # c0 +=  a013  * b13 */
                       "vfmadd.s vv1,  vv29, vs18, vv1 \n\t"  /* # c1 +=  a113  * b13 */
                       "vfmadd.s vv2,  vv29, vs19, vv2 \n\t"  /* # c2 +=  a213  * b13 */
                       "vfmadd.s vv3,  vv29, vs20, vv3 \n\t"  /* # c3 +=  a313  * b13 */
                       "vfmadd.s vv4,  vv29, vs21, vv4 \n\t"  /* # c4 +=  a413  * b13 */
                       "vfmadd.s vv5,  vv29, vs22, vv5 \n\t"  /* # c5 +=  a513  * b13 */
                       "vfmadd.s vv6,  vv29, vs23, vv6 \n\t"  /* # c6 +=  a613  * b13 */
                       "vfmadd.s vv7,  vv29, vs24, vv7 \n\t"  /* # c7 +=  a713  * b13 */
                       "vfmadd.s vv8,  vv29, vs25, vv8 \n\t"  /* # c8 +=  a813  * b13 */
                       "vfmadd.s vv9,  vv29, vs26, vv9 \n\t"  /* # c9 +=  a913  * b13 */
                       "vfmadd.s vv10, vv29, vs27, vv10\n\t"  /* # c10 += a1013 * b13 */
                       "vfmadd.s vv11, vv29, vs28, vv11\n\t"  /* # c11 += a1113 * b13 */
                       "vfmadd.s vv12, vv29, vs29, vv12\n\t"  /* # c12 += a1213 * b13 */
                       "vfmadd.s vv13, vv29, vs30, vv13\n\t"  /* # c13 += a1313 * b13 */
                       "vfmadd.s vv14, vv29, vs31, vv14\n\t"  /* # c14 += a1413 * b13 */
                       "vfmadd.s vv15, vv29, vs32, vv15\n\t"  /* # c15 += a1513 * b13 */
                       "vfmadd.s vv0,  vv30, vs33, vv0 \n\t"  /* # c0 +=  a014  * b14 */
                       "vfmadd.s vv1,  vv30, vs34, vv1 \n\t"  /* # c1 +=  a114  * b14 */
                       "vfmadd.s vv2,  vv30, vs35, vv2 \n\t"  /* # c2 +=  a214  * b14 */
                       "vfmadd.s vv3,  vv30, vs36, vv3 \n\t"  /* # c3 +=  a314  * b14 */
                       "vfmadd.s vv4,  vv30, vs37, vv4 \n\t"  /* # c4 +=  a414  * b14 */
                       "vfmadd.s vv5,  vv30, vs38, vv5 \n\t"  /* # c5 +=  a514  * b14 */
                       "vfmadd.s vv6,  vv30, vs39, vv6 \n\t"  /* # c6 +=  a614  * b14 */
                       "vfmadd.s vv7,  vv30, vs40, vv7 \n\t"  /* # c7 +=  a714  * b14 */
                       "vfmadd.s vv8,  vv30, vs41, vv8 \n\t"  /* # c8 +=  a814  * b14 */
                       "vfmadd.s vv9,  vv30, vs42, vv9 \n\t"  /* # c9 +=  a914  * b14 */
                       "vfmadd.s vv10, vv30, vs43, vv10\n\t"  /* # c10 += a1014 * b14 */
                       "vfmadd.s vv11, vv30, vs44, vv11\n\t"  /* # c11 += a1114 * b14 */
                       "vfmadd.s vv12, vv30, vs45, vv12\n\t"  /* # c12 += a1214 * b14 */
                       "vfmadd.s vv13, vv30, vs46, vv13\n\t"  /* # c13 += a1314 * b14 */
                       "vfmadd.s vv14, vv30, vs47, vv14\n\t"  /* # c14 += a1414 * b14 */
                       "vfmadd.s vv15, vv30, vs48, vv15\n\t"  /* # c15 += a1514 * b14 */
 
                       "vstop                          \n\t" : : );

          __asm__ volatile(".align 3                       \n\t"
                       "hwacha_sgemm_internal_main_p6: \n\t"
                       "vfmadd.s vv0,  vv31, vs1, vv0  \n\t"  /* # c0 +=  a015  * b15 */
                       "vfmadd.s vv1,  vv31, vs2, vv1  \n\t"  /* # c1 +=  a115  * b15 */
                       "vfmadd.s vv2,  vv31, vs3, vv2  \n\t"  /* # c2 +=  a215  * b15 */
                       "vfmadd.s vv3,  vv31, vs4, vv3  \n\t"  /* # c3 +=  a315  * b15 */
                       "vfmadd.s vv4,  vv31, vs5, vv4  \n\t"  /* # c4 +=  a415  * b15 */
                       "vfmadd.s vv5,  vv31, vs6, vv5  \n\t"  /* # c5 +=  a515  * b15 */
                       "vfmadd.s vv6,  vv31, vs7, vv6  \n\t"  /* # c6 +=  a615  * b15 */
                       "vfmadd.s vv7,  vv31, vs8, vv7  \n\t"  /* # c7 +=  a715  * b15 */
                       "vfmadd.s vv8,  vv31, vs9, vv8  \n\t"  /* # c8 +=  a815  * b15 */
                       "vfmadd.s vv9,  vv31, vs10, vv9 \n\t"  /* # c9 +=  a915  * b15 */
                       "vfmadd.s vv10, vv31, vs11, vv10\n\t"  /* # c10 += a1015 * b15 */
                       "vfmadd.s vv11, vv31, vs12, vv11\n\t"  /* # c11 += a1115 * b15 */
                       "vfmadd.s vv12, vv31, vs13, vv12\n\t"  /* # c12 += a1215 * b15 */
                       "vfmadd.s vv13, vv31, vs14, vv13\n\t"  /* # c13 += a1315 * b15 */
                       "vfmadd.s vv14, vv31, vs15, vv14\n\t"  /* # c14 += a1415 * b15 */
                       "vfmadd.s vv15, vv31, vs16, vv15\n\t"  /* # c15 += a1515 * b15 */
 
                       "vstop                          \n\t" : : );


          //tail handling: load A, B and matmul vector fetch block
          __asm__ volatile(".align 3                   \n\t"
                       "hwacha_sgemm_internal_tail:   \n\t"
                       "vlw vv16, va16                 \n\t"   /* # b0             */
                       "vfmul.s vv16, vv16, vs62       \n\t"   /* # b0 * alpha     */
                       "vfmadd.s vv0, vv16, vs1, vv0   \n\t"   /* # c0 += a00 * b0 */
                       "vfmadd.s vv1, vv16, vs2, vv1   \n\t"   /* # c0 += a10 * b0 */
                       "vfmadd.s vv2, vv16, vs3, vv2   \n\t"   /* # c0 += a20 * b0 */
                       "vfmadd.s vv3, vv16, vs4, vv3   \n\t"   /* # c0 += a30 * b0 */
                       "vfmadd.s vv4, vv16, vs5, vv4   \n\t"   /* # c0 += a40 * b0 */
                       "vfmadd.s vv5, vv16, vs6, vv5   \n\t"   /* # c0 += a50 * b0 */
                       "vfmadd.s vv6, vv16, vs7, vv6   \n\t"   /* # c0 += a60 * b0 */
                       "vfmadd.s vv7, vv16, vs8, vv7   \n\t"   /* # c0 += a70 * b0 */
                       "vfmadd.s vv8, vv16, vs9, vv8   \n\t"   /* # c0 += a80 * b0 */
                       "vfmadd.s vv9, vv16, vs10, vv9  \n\t"   /* # c0 += a90 * b0 */
                       "vfmadd.s vv10, vv16, vs11, vv10  \n\t"   /* # c0 += a100 * b0 */
                       "vfmadd.s vv11, vv16, vs12, vv11  \n\t"   /* # c0 += a110 * b0 */
                       "vfmadd.s vv12, vv16, vs13, vv12  \n\t"   /* # c0 += a120 * b0 */
                       "vfmadd.s vv13, vv16, vs14, vv13  \n\t"   /* # c0 += a130 * b0 */
                       "vfmadd.s vv14, vv16, vs15, vv14  \n\t"   /* # c0 += a140 * b0 */
                       "vfmadd.s vv15, vv16, vs16, vv15  \n\t"   /* # c0 += a150 * b0 */
                       "vstop                          \n\t" : : );


          //store C vector fetch block 
          __asm__ volatile(".align 3                       \n\t"
                       "hwacha_sgemm_internal_post:     \n\t"
                       "vsw vv0,  va0                  \n\t"
                       "vsw vv1,  va1                  \n\t"
                       "vsw vv2,  va2                  \n\t"
                       "vsw vv3,  va3                  \n\t"
                       "vsw vv4,  va4                  \n\t"
                       "vsw vv5,  va5                  \n\t"
                       "vsw vv6,  va6                  \n\t"
                       "vsw vv7,  va7                  \n\t"
                       "vsw vv8,  va8                  \n\t"
                       "vsw vv9,  va9                  \n\t"
                       "vsw vv10, va10                 \n\t"
                       "vsw vv11, va11                 \n\t"
                       "vsw vv12, va12                 \n\t"
                       "vsw vv13, va13                 \n\t"
                       "vsw vv14, va14                 \n\t"
                       "vsw vv15, va15                 \n\t"
                       "vstop                          \n\t" : :);

          __asm__ volatile("end_vf:                        \n\t"
                           "fence                          \n\t" ::);

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
