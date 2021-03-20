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
	.global bli_1v_hwacha_vf_init
bli_1v_hwacha_vf_init:
	vpset vp0
	vstop

	.align 3
	.global bli_saxpbyv_unit_hwacha_vf_main
bli_saxpbyv_unit_hwacha_vf_main:
	/* load elements of y */
	OP vlw vv0,, va0,,
	/* y = beta * y */
	OP vfmul.s vv2,, vv0,, vs2
	/* load elements of x */
	OP vlw vv1,, va1,,
	/* y = y + alpha * x */
	OP vfmadd.s vv2,, vv1,, vs1,, vv2
	OP vsw vv2,, va0,,
	vstop

	.align 3
	.global bli_saxpbyv_stride_hwacha_vf_main
bli_saxpbyv_stride_hwacha_vf_main:
	/* load elements of y */
	OP vlstw vv0,, va0,, va2
	/* y = beta * y */
	OP vfmul.s vv2,, vv0,, vs2
	/* load elements of x */
	OP vlstw vv1,, va1,, va3
	/* y = y + alpha * x */
	OP vfmadd.s vv2,, vv1,, vs1,, vv2
	OP vsstw vv2,, va0,, va2
	vstop

	.align 3
	.global bli_saxpyv_unit_hwacha_vf_main
bli_saxpyv_unit_hwacha_vf_main:
	/* load elements of y */
	OP vlw vv0,, va0,,
	/* load elements of x */
	OP vlw vv1,, va1,,
	/* y = y + alpha * x */
	OP vfmadd.s vv0,, vv1,, vs1,, vv0
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_saxpyv_stride_hwacha_vf_main
bli_saxpyv_stride_hwacha_vf_main:
	/* load elements of y */
	OP vlstw vv0,, va0,, va2
	/* load elements of x */
	OP vlstw vv1,, va1,, va3
	/* y = y + alpha * x */
	OP vfmadd.s vv0,, vv1,, vs1,, vv0
	OP vsstw vv0,, va0,, va2
	vstop

	.align 3
	.global bli_saddv_unit_hwacha_vf_main
bli_saddv_unit_hwacha_vf_main:
	/* load elements of y */
	OP vlw vv0,, va0,,
	/* load elements of x */
	OP vlw vv1,, va1,,
	/* y = y + x */
	OP vfadd.s vv0,, vv1,, vv0
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_saddv_stride_hwacha_vf_main
bli_saddv_stride_hwacha_vf_main:
	/* load elements of y */
	OP vlstw vv0,, va0,, va2
	/* load elements of x */
	OP vlstw vv1,, va1,, va3
	/* y = y + x */
	OP vfadd.s vv0,, vv1,, vv0
	OP vsstw vv0,, va0,, va2
	vstop
