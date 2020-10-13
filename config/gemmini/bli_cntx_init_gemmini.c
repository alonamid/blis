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
//need to make sure that all variable declarations in gemmini_params.h
//are static const, otherwise there will be linker issues


void bli_cntx_init_gemmini( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_gemmini_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  //0,
	  5,
	  // gemm
	  //BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_gemmini_small_os,            TRUE,
	  BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_gemmini_small_ws,            TRUE,
          //trsm
          BLIS_TRSM_U_UKR,     BLIS_FLOAT,    bli_strsm_u_gemmini_small,             TRUE,
          BLIS_TRSM_L_UKR,     BLIS_FLOAT,    bli_strsm_l_gemmini_small,             TRUE,
          //gemmtrsm
          //BLIS_GEMMTRSM_U_UKR, BLIS_FLOAT,    bli_sgemmtrsm_u_gemmini_small_os,      TRUE,
          //BLIS_GEMMTRSM_L_UKR, BLIS_FLOAT,    bli_sgemmtrsm_l_gemmini_small_os,      TRUE,
          BLIS_GEMMTRSM_U_UKR, BLIS_FLOAT,    bli_sgemmtrsm_u_gemmini_small_ws,      TRUE,
          BLIS_GEMMTRSM_L_UKR, BLIS_FLOAT,    bli_sgemmtrsm_l_gemmini_small_ws,      TRUE,
	  cntx
	);

	// Update the context with optimized level-1f kernels.
	bli_cntx_set_l1f_kers
	(
	  0,
	  cntx
	);

	// Update the context with optimized level-1v kernels.
	bli_cntx_set_l1v_kers
	(
	  0,
	  cntx
	);

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)
#define L2_SIZE 512*1024
#define l2_elem_capacity (L2_SIZE / sizeof(elem_t))
#define max_tile_l2 (l2_elem_capacity / max_tile_k)

	// Update the context with optimized packm kernels.
        bli_cntx_set_packm_kers
        (
          1,
          //BLIS_PACKM_32XK_KER,  BLIS_FLOAT, bli_spackm_gemmini_cxk,
          max_tile_i_j,  BLIS_FLOAT, bli_spackm_gemmini_32xk,
          //max_tile_i_j,  BLIS_FLOAT, bli_spackm_gemmini_cxk,
          cntx
        );

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                               s      d      c      z
        //register blocking (array size)
        // OS
	//bli_blksz_init_easy( &blkszs[ BLIS_MR ],         DIM,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_NR ],         DIM,     0,     0,     0 );
        // WS
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],         max_tile_i_j,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],         max_tile_i_j,     0,     0,     0 );

        //cache blocking (scratchpad size)
        //TODO (Alon): Consider blocking based on L2 size rather than scratchpad size?
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],max_tile_i_j,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],  max_tile_k,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],max_tile_l2,     0,     0,     0 );

	// level-1f
	//bli_blksz_init_easy( &blkszs[ BLIS_AF ],         0,     0,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_DF ],         0,     0,     0,     0 );

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k

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
}

