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
	.global bli_1v_hwacha_vf_hinit
bli_1v_hwacha_vf_hinit:
	vpset vp0
	vfcvt.h.s vs1, vs1
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
	.global bli_sxpbyv_unit_hwacha_vf_main
bli_sxpbyv_unit_hwacha_vf_main:
	/* load elements of y */
	OP vlw vv0,, va0,,
	/* load elements of x */
	OP vlw vv1,, va1,,
	/* y = beta * y + x */
	OP vfmadd.s vv0,, vv0,, vs1,, vv1
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_sxpbyv_stride_hwacha_vf_main
bli_sxpbyv_stride_hwacha_vf_main:
	/* load elements of y */
	OP vlstw vv0,, va0,, va2
	/* load elements of x */
	OP vlstw vv1,, va1,, va3
	/* y = beta * y + x */
	OP vfmadd.s vv0,, vv0,, vs1,, vv1
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

	.align 3
	.global bli_ssubv_unit_hwacha_vf_main
bli_ssubv_unit_hwacha_vf_main:
	/* load elements of y */
	OP vlw vv0,, va0,,
	/* load elements of x */
	OP vlw vv1,, va1,,
	/* y = y + x */
	OP vfsub.s vv0,, vv0,, vv1
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_ssubv_stride_hwacha_vf_main
bli_ssubv_stride_hwacha_vf_main:
	/* load elements of y */
	OP vlstw vv0,, va0,, va2
	/* load elements of x */
	OP vlstw vv1,, va1,, va3
	/* y = y + x */
	OP vfsub.s vv0,, vv0,, vv1
	OP vsstw vv0,, va0,, va2
	vstop

	.align 3
	.global bli_sswapv_unit_hwacha_vf_main
bli_sswapv_unit_hwacha_vf_main:
	/* load elements of y */
	OP vlw vv0,, va0,,
	/* load elements of x */
	OP vlw vv1,, va1,,
	OP vsw vv1,, va0,,
	OP vsw vv0,, va1,,
	vstop

	.align 3
	.global bli_sswapv_stride_hwacha_vf_main
bli_sswapv_stride_hwacha_vf_main:
	/* load elements of y */
	OP vlstw vv0,, va0,, va2
	/* load elements of x */
	OP vlstw vv1,, va1,, va3
	OP vsstw vv1,, va0,, va2
	OP vsstw vv0,, va1,, va3
	vstop

	.align 3
	.global bli_scopyv_unit_hwacha_vf_main
bli_scopyv_unit_hwacha_vf_main:
	/* load elements of x */
	OP vlw vv0,, va1,,
	/* store into y */
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_scopyv_stride_hwacha_vf_main
bli_scopyv_stride_hwacha_vf_main:
	/* load elements of x */
	OP vlstw vv0,, va1,, va3
	/* store into y */
	OP vsstw vv0,, va0,, va2
	vstop

	.align 3
	.global bli_scopyconvertv_unit_hwacha_vf_main
bli_scopyconvertv_unit_hwacha_vf_main:
	/* load elements of x */
	OP vlw vv0,, va1,,
	/* convert */
	OP vfcvt.h.s vv1,, vv0,,
	/* store into y */
	OP vsh vv1,, va0,,
	vstop

	.align 3
	.global bli_scopyconvertv_stride_hwacha_vf_main
bli_scopyconvertv_stride_hwacha_vf_main:
	/* load elements of x */
	OP vlstw vv0,, va1,, va3
	/* convert */
	OP vfcvt.h.s vv1,, vv0,,
	/* store into y */
	OP vssth vv1,, va0,, va2
	vstop

	.align 3
	.global bli_ssetv_unit_hwacha_vf_main
bli_ssetv_unit_hwacha_vf_main:
	/* set elements of x */
	OP vfadd.s vv0,, vs0,, vs1,,
	/* store into x */
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_ssetv_stride_hwacha_vf_main
bli_ssetv_stride_hwacha_vf_main:
	/* set elements of x */
	OP vfadd.s vv0,, vs0,, vs1,,
	/* store into x */
	OP vsstw vv0,, va0,, va1
	vstop

	.align 3
	.global bli_hsetv_unit_hwacha_vf_main
bli_hsetv_unit_hwacha_vf_main:
	/* set elements of x */
	OP vfcvt.h.s vv0,, vs1,,
	/* store into x */
	OP vsh vv0,, va0,,
	vstop

	.align 3
	.global bli_hsetv_stride_hwacha_vf_main
bli_hsetv_stride_hwacha_vf_main:
	/* set elements of x */
	OP vfcvt.h.s vv0,, vs1,,
	/* store into x */
	OP vssth vv0,, va0,, va1
	vstop

	.align 3
	.global bli_sscalv_unit_hwacha_vf_main
bli_sscalv_unit_hwacha_vf_main:
	/* load x */
	OP vlw vv0,, va0,,
	/* scale elements of x */
	OP vfmul.s vv0,, vv0,, vs1,,
	/* store into x */
	OP vsw vv0,, va0,,
	vstop

	.align 3
	.global bli_sscalv_stride_hwacha_vf_main
bli_sscalv_stride_hwacha_vf_main:
	/* load x */
	OP vlstw vv0,, va0,, va1
	/* set elements of x */
	OP vfmul.s vv0,, vv0,, vs1,,
	/* store into x */
	OP vsstw vv0,, va0,, va1
	vstop

	.align 3
	.global bli_hscalv_unit_hwacha_vf_main
bli_hscalv_unit_hwacha_vf_main:
	/* load x */
	OP vlh vv0,, va0,,
	/* scale elements of x */
	OP vfmul.h vv0,, vv0,, vs1,,
	/* store into x */
	OP vsh vv0,, va0,,
	vstop

	.align 3
	.global bli_hscalv_stride_hwacha_vf_main
bli_hscalv_stride_hwacha_vf_main:
	/* load x */
	OP vlstw vv0,, va0,, va1
	/* scale elements of x */
	OP vfmul.h vv0,, vv0,, vs1,,
	/* store into x */
	OP vssth vv0,, va0,, va1
	vstop
