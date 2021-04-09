/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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
#include "include/gemmini_params.h"
#include <sys/mman.h>
//need to make sure that all variable declarations in gemmini_params.h
//are static const, otherwise there will be linker issues


void bli_cntx_init_gemminihwacha( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
        blksz_t thresh[ BLIS_NUM_THRESH ];

        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
          perror("mlockall failed");
          exit(1);
        }

	// Set default kernel blocksizes and functions.
	bli_cntx_init_gemminihwacha_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  5,
	  // gemm
	  BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_gemmini_fsm_ws,            TRUE,
	  //trsm
	  BLIS_TRSM_U_UKR,     BLIS_FLOAT,    bli_strsm_u_gemmini_small,             TRUE,
	  BLIS_TRSM_L_UKR,     BLIS_FLOAT,    bli_strsm_l_gemmini_small,             TRUE,
	  //gemmtrsm
	  BLIS_GEMMTRSM_U_UKR, BLIS_FLOAT,    bli_sgemmtrsm_u_gemmini_fsm_ws,      TRUE,
	  BLIS_GEMMTRSM_L_UKR, BLIS_FLOAT,    bli_sgemmtrsm_l_gemmini_fsm_ws,      TRUE,
	  cntx
	);

	// Update the context with optimized level-1f kernels.
	bli_cntx_set_l1f_kers
	(
	  //0,
	  4,
	  BLIS_DOTXF_KER,  BLIS_FLOAT, bli_sdotxf_hwacha,
	  BLIS_AXPYF_KER,  BLIS_FLOAT, bli_saxpyf_hwacha,
	  BLIS_AXPY2V_KER,  BLIS_FLOAT, bli_saxpy2v_hwacha,
	  BLIS_DOTAXPYV_KER,  BLIS_FLOAT, bli_sdotaxpyv_hwacha,
	  cntx
	);

	// Update the context with optimized level-1v kernels.
	bli_cntx_set_l1v_kers
	(
	  5,
          BLIS_INVERTV_KER,  BLIS_FLOAT, bli_sinvertv_lowprec,
          BLIS_SETV_KER,  BLIS_FLOAT, bli_ssetv_lowprec,
          BLIS_SCALV_KER,  BLIS_FLOAT, bli_sscalv_lowprec,
          BLIS_SCAL2V_KER,  BLIS_FLOAT, bli_sscal2v_lowprec,
          BLIS_COPYV_KER,  BLIS_FLOAT, bli_scopyv_lowprec,
	  //0,
	  cntx
	);


//single buffering
/*
#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)
*/
//double buffering use half the memory resources
#define partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc ((ACC_ROWS / 2) / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

//L2 parameters
#define L2_SIZE 512*1024
#define l2_elem_capacity (L2_SIZE / sizeof(elem_t))
#define max_tile_l2 (l2_elem_capacity * max_tile_i_j / mats_in_partition)

	// Update the context with optimized packm kernels.
        bli_cntx_set_packm_kers
        (
          //0,
          3,
          //BLIS_PACKM_4XK_KER,   BLIS_FLOAT, bli_spackm_gemmini_4xk,
          //BLIS_PACKM_32XK_KER,  BLIS_FLOAT, bli_spackm_gemmini_32xk,
          //BLIS_PACKM_88XK_KER,  BLIS_FLOAT, bli_spackm_gemmini_88xk,
          //DIM*max_tile_i_j,  BLIS_FLOAT, bli_spackm_gemmini_cxk,
          //BLIS_PACKM_64XK_KER,  BLIS_FLOAT, bli_spackm_gemmini_cxk,
          //hwacha based packing
          BLIS_PACKM_4XK_KER,   BLIS_FLOAT, bli_spackm_hwacha_cxk,
          BLIS_PACKM_32XK_KER,  BLIS_FLOAT, bli_spackm_hwacha_cxk,
          DIM*max_tile_i_j,  BLIS_FLOAT, bli_spackm_hwacha_cxk,
          cntx
        );

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                               s      d      c      z
        //register blocking (array size)
        // OS
	//bli_blksz_init_easy( &blkszs[ BLIS_MR ],         DIM,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_NR ],         DIM,     0,     0,     0 );
        // WS
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],         DIM*max_tile_i_j,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],         DIM*max_tile_i_j,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_MR ],         64,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_NR ],         64,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_MR ],         DIM,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_NR ],         DIM,     0,     0,     0 );

        //cache blocking (scratchpad size)
        //TODO (Alon): Consider blocking based on L2 size rather than scratchpad size?
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],DIM*max_tile_i_j,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],4*DIM*max_tile_k,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],4*max_tile_l2,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_MC ],64,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_KC ],4*DIM*max_tile_k,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_NC ],4*max_tile_l2,     0,     0,     0 );

	// level-1f
	//bli_blksz_init_easy( &blkszs[ BLIS_AF ],         0,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_DF ],         0,     0,     0,     0 );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  //BLIS_NAT, 0,
	  //BLIS_NAT, 7,
	  BLIS_NAT, 5,
	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
	  // level-1f
	  //BLIS_AF, &blkszs[ BLIS_AF ], BLIS_AF,
	  //BLIS_DF, &blkszs[ BLIS_DF ], BLIS_DF,
	  cntx
	);


	// small matrix:

	// Initialize sup thresholds with architecture-appropriate values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &thresh[ BLIS_MT ],   2*DIM*DIM,   -1,    -1,    -1 );
	bli_blksz_init_easy( &thresh[ BLIS_NT ],   2*DIM*DIM,   -1,    -1,    -1 );
	bli_blksz_init_easy( &thresh[ BLIS_KT ],   10,  -1,    -1,    -1 );
	//bli_blksz_init_easy( &thresh[ BLIS_KT ],   DIM*max_tile_i_j / 2,   220,    -1,    -1 );
	//bli_blksz_init_easy( &thresh[ BLIS_MT ],   20,   256,    -1,    -1 );
	//bli_blksz_init_easy( &thresh[ BLIS_NT ],   20,   256,    -1,    -1 );
	//bli_blksz_init_easy( &thresh[ BLIS_KT ],   20,   220,    -1,    -1 );

	// Initialize the context with the sup thresholds.
	bli_cntx_set_l3_sup_thresh
	(
	  3,
	  BLIS_MT, &thresh[ BLIS_MT ],
	  BLIS_NT, &thresh[ BLIS_NT ],
	  BLIS_KT, &thresh[ BLIS_KT ],
	  cntx
	);

	// Initialize the context with the sup handlers.
	bli_cntx_set_l3_sup_handlers
	(
	  1,
	  BLIS_GEMM, bli_gemmsup_ref,
	  //BLIS_GEMMT, bli_gemmtsup_ref,
	  cntx
	);


	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                           s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR ],     6,     6,    -1,    -1,
	                                             9,     9,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    16,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,    72,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   256,   256,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8160,  4080,    -1,    -1 );


	// Update the context with the current architecture's register and cache
	// blocksizes for small/unpacked level-3 problems.
	bli_cntx_set_l3_sup_blkszs
	(
	  5,
	  BLIS_NC, &blkszs[ BLIS_NC ],
	  BLIS_KC, &blkszs[ BLIS_KC ],
	  BLIS_MC, &blkszs[ BLIS_MC ],
	  BLIS_NR, &blkszs[ BLIS_NR ],
	  BLIS_MR, &blkszs[ BLIS_MR ],
	  cntx
	);
#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k

}

