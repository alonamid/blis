/* SPDX-License-Identifier: BSD-3-Clause */

	.set MR, 16
	.set NR, 16

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
	vsw vv0, va16
        vstop

	.align 3
	.set i, 1
	.rept (MR-1)
	LABELGEN bli_sgemmtrsm_l_hwacha_16xn_vf_inner_, %i
        /* load B[i][_] */
        /* workspace_vector = 0 */
	OP vfsub.s vv16,, vv16,, vv16,,
	.set k, 0
	.rept i
	/* workspace_vector += A[i][k] * B[k][_] */
	OP vfmadd.s vv16,, vv,%k, vs,%(k+2), vv16,,
	.set k, k+1
	.endr
        /* workspace_vector = B[i][_] - workspace_vector */
	OP vfsub.s vv16,, vv,%i, vv16,,
        /* B[i][_] = A[i][i] * workspace_vector */
	OP vfmul.s vv,%i, vv16,, vs1,,
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
	vfmul.s vv15, vv15, vs1
	vsw vv15, va15
	vsw vv15, va16
        vstop

	.align 3
	.set i, 1
	.rept (MR-1)
	LABELGEN bli_sgemmtrsm_u_hwacha_16xn_vf_inner_, %i
        /* load B[i][_] */
        /* workspace_vector = 0 */
	OP vfsub.s vv16,, vv16,, vv16,,
	.set k, 0
	.rept i
	/* workspace_vector += A[i][k] * B[k][_] */
	OP vfmadd.s vv16,, vv,%(MR-1-k), vs,%(k+2), vv16,,
	.set k, k+1
	.endr
        /* workspace_vector = B[i][_] - workspace_vector */
	OP vfsub.s vv16,, vv,%(MR-1-i), vv16,,
        /* B[i][_] = A[i][i] * workspace_vector */
	OP vfmul.s vv,%(MR-1-i), vv16,, vs1,,
	OP vsw vv,%(MR-1-i), va,%(MR-1-i)
        /* C[i][_] = B[i][_] */
	OP vsw vv,%(MR-1-i), va,%(i+MR)
	vstop
	.set i, i+1
	.endr
