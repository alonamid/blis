/* SPDX-License-Identifier: BSD-3-Clause */

	.set MR, 16
	.set NR, 32

	.text

	.altmacro

	.macro OP insn, r0, n0, r1, n1, r2, n2, r3, n3
	.ifb r1
	\insn \r0\n0
	.else
	.ifb r2
	\insn \r0\n0, \r1\n1
	.else
	.ifb r3
	\insn \r0\n0, \r1\n1, \r2\n2
	.else
	\insn \r0\n0, \r1\n1, \r2\n2, \r3\n3
	.endif
	.endif
	.endif
	.endm


	.macro LABELGEN name, n0
	.global \name\n0
	 \name\n0:
	.endm

	.align 3
	.global bli_sgemmtrsm_l_hwacha_16xn_vf_inner_0
bli_sgemmtrsm_l_hwacha_16xn_vf_inner_0:
        vpset vp0
	vfmul.s vv0, vv0, vs1
	vsw vv0, va0
	OP vsw vv0,, va,%(MR)
        vstop

	.align 3
	.set i, 1
	.rept (MR-1)
	LABELGEN bli_sgemmtrsm_l_hwacha_16xn_vf_inner_, %i
        /* load B[i][_] */
        /* workspace_vector = 0 */
	OP vfsub.s vv,%(MR), vv,%(MR), vv,%(MR),
	.set k, 0
	.rept i
	/* workspace_vector += A[i][k] * B[k][_] */
	OP vfmadd.s vv,%(MR), vv,%k, vs,%(k+2), vv,%(MR),
	.set k, k+1
	.endr
        /* workspace_vector = B[i][_] - workspace_vector */
	OP vfsub.s vv,%(MR), vv,%i, vv,%(MR),
        /* B[i][_] = A[i][i] * workspace_vector */
	OP vfmul.s vv,%i, vv,%(MR), vs1,,
	OP vsw vv,%(i), va,%(i)
        /* C[i][_] = B[i][_] */
	OP vsw vv,%(i), va,%(i+MR)
	vstop
	.set i, i+1
	.endr


	.align 3
	.global bli_sgemmtrsm_u_hwacha_16xn_vf_inner_0
bli_sgemmtrsm_u_hwacha_16xn_vf_inner_0:
        vpset vp0
	OP vfmul.s vv,%(MR-1), vv,%(MR-1), vs1,,
	OP vsw vv, %(MR-1), va, %(MR-1)
	OP vsw vv,%(MR-1), va,%(MR)
        vstop

	.align 3
	.set i, 1
	.rept (MR-1)
	LABELGEN bli_sgemmtrsm_u_hwacha_16xn_vf_inner_, %i
        /* load B[i][_] */
        /* workspace_vector = 0 */
	OP vfsub.s vv,%(MR), vv,%(MR), vv,%(MR)
	.set k, 0
	.rept i
	/* workspace_vector += A[i][k] * B[k][_] */
	OP vfmadd.s vv,%(MR), vv,%(MR-1-k), vs,%(k+2), vv,%(MR)
	.set k, k+1
	.endr
        /* workspace_vector = B[i][_] - workspace_vector */
	OP vfsub.s vv,%(MR), vv,%(MR-1-i), vv,%(MR)
        /* B[i][_] = A[i][i] * workspace_vector */
	OP vfmul.s vv,%(MR-1-i), vv,%(MR), vs1,,
	OP vsw vv,%(MR-1-i), va,%(MR-1-i)
        /* C[i][_] = B[i][_] */
	OP vsw vv,%(MR-1-i), va,%(i+MR)
	vstop
	.set i, i+1
	.endr



	.align 3
	.global bli_sgemmtrsm_l_hwacha_16xn_vf_inner_cmajor_0
bli_sgemmtrsm_l_hwacha_16xn_vf_inner_cmajor_0:
        vpset vp0
	OP vfmul.s vv0,, vv0,, vs1,,
	OP vsw vv0,, va0,,
	OP vsstw vv0,, va,%(MR), va31
        vstop

	.align 3
	.global bli_sgemmtrsm_u_hwacha_16xn_vf_inner_cmajor_0
bli_sgemmtrsm_u_hwacha_16xn_vf_inner_cmajor_0:
        vpset vp0
	OP vfmul.s vv,%(MR-1), vv,%(MR-1), vs1,,
	OP vsw vv, %(MR-1), va, %(MR-1),,
	OP vsstw vv,%(MR-1), va,%(MR), va31,,
        vstop

	.align 3
	.set i, 1
	.rept (MR-1)
	LABELGEN bli_sgemmtrsm_l_hwacha_16xn_vf_inner_cmajor_, %i
        /* load B[i][_] */
        /* workspace_vector = 0 */
	OP vfsub.s vv,%(MR), vv,%(MR), vv,%(MR),
	.set k, 0
	.rept i
	/* workspace_vector += A[i][k] * B[k][_] */
	OP vfmadd.s vv,%(MR), vv,%k, vs,%(k+2), vv,%(MR),
	.set k, k+1
	.endr
        /* workspace_vector = B[i][_] - workspace_vector */
	OP vfsub.s vv,%(MR), vv,%i, vv,%(MR),
        /* B[i][_] = A[i][i] * workspace_vector */
	OP vfmul.s vv,%i, vv,%(MR), vs1,,
	OP vsw vv,%(i), va,%(i)
        /* C[i][_] = B[i][_] */
	OP vsstw vv,%(i), va,%(i+MR), va,0
	vstop
	.set i, i+1
	.endr



	.align 3
	.set i, 1
	.rept (MR-1)
	LABELGEN bli_sgemmtrsm_u_hwacha_16xn_vf_inner_cmajor_, %i
        /* load B[i][_] */
        /* workspace_vector = 0 */
	OP vfsub.s vv,%(MR), vv,%(MR), vv,%(MR)
	.set k, 0
	.rept i
	/* workspace_vector += A[i][k] * B[k][_] */
	OP vfmadd.s vv,%(MR), vv,%(MR-1-k), vs,%(k+2), vv,%(MR)
	.set k, k+1
	.endr
        /* workspace_vector = B[i][_] - workspace_vector */
	OP vfsub.s vv,%(MR), vv,%(MR-1-i), vv,%(MR)
        /* B[i][_] = A[i][i] * workspace_vector */
	OP vfmul.s vv,%(MR-1-i), vv,%(MR), vs1,,
	OP vsw vv,%(MR-1-i), va,%(MR-1-i)
        /* C[i][_] = B[i][_] */
	OP vsstw vv,%(MR-1-i), va,%(i+MR), va,%(MR-1)
	vstop
	.set i, i+1
	.endr
