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
#include <sys/mman.h>

void bli_cntx_init_hwacha( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
        blksz_t thresh[ BLIS_NUM_THRESH ];

        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
          perror("mlockall failed");
          exit(1);
        }

	// Set default kernel blocksizes and functions.
	bli_cntx_init_hwacha_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  //0,
	  1,
	  // gemm
	  BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_hwacha_16x16,            TRUE,
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

	// Update the context with optimized packm kernels.
        bli_cntx_set_packm_kers
        (
          0,
	  cntx
        );

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                               s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],        16,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],        16,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],        256,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],        512,     0,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],        512,     0,     0,     0 );

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
	bli_blksz_init_easy( &thresh[ BLIS_MT ],     8,   -1,    -1,    -1 );
	bli_blksz_init_easy( &thresh[ BLIS_NT ],     8,   -1,    -1,    -1 );
	bli_blksz_init_easy( &thresh[ BLIS_KT ],     8,  -1,    -1,    -1 );
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

}

