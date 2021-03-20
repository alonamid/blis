/* SPDX-License-Identifier: BSD-3-Clause */

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

	.align 3
	.global bli_sdotxf_hwacha_vf_init
bli_sdotxf_hwacha_vf_init:
	vpset vp0
	OP vadd vv0,, vs0,, vs0
	OP vadd vv3,, vs0,, vs0
	vstop

	.align 3
	.global bli_sdotxf_hwacha_vf_init_beta
bli_sdotxf_hwacha_vf_init_beta:
	vpset vp0
	OP vlw vv1,, va0,,
	OP vfmul.s vv0,, vv1,, vs1
	OP vadd vv3,, vs0,, vs0
	vstop

	.align 3
	.global bli_sdotxf_hwacha_vf_main
bli_sdotxf_hwacha_vf_main:
	/* load elements of A */
	OP vlstw vv2,, va1,, va2,,
	/* y = A[:,m] * x[m] */
	OP vfmadd.s vv3,, vv2,, vs2,, vv3
	vstop

	.align 3
	.global bli_sdotxf_hwacha_vf_alpha
bli_sdotxf_hwacha_vf_alpha:
	/* y = alpha * A[:,m] * x[m] */
	OP vfmadd.s vv0,, vv3,, vs3,, vv0 
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_sdotxf_hwacha_vf_early_end
bli_sdotxf_hwacha_vf_early_end:
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_saxpyf_hwacha_vf_init
bli_saxpyf_hwacha_vf_init:
	vpset vp0
	OP vlw vv0,, va0,,
	OP vadd vv2,, vs0,, vs0
	vstop

	.align 3
	.global bli_saxpyf_hwacha_vf_main
bli_saxpyf_hwacha_vf_main:
	/* load elements of A */
	OP vlw vv1,, va1,,
	/* y = A[:,j] * x[j] */
	OP vfmadd.s vv2,, vv1,, vs1,, vv2
	vstop

	.align 3
	.global bli_saxpyf_hwacha_vf_end
bli_saxpyf_hwacha_vf_end:
	/* y = alpha * y[] */
	OP vfmadd.s vv0,, vv2,, vs2,, vv0
	OP vsw vv0,, va0,,
	vstop
