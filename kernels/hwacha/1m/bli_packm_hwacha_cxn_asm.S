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
	.global bli_packm_hwacha_vf_init
bli_packm_hwacha_vf_init:
	vpset vp0
	vstop

	.align 3
	.global bli_packm_hwacha_vf_sset0
bli_packm_hwacha_vf_sset0:
	OP vadd vv2,, vs0,, vs0
	OP vsh vv2,, va0,
	vstop

	.align 3
	.global bli_packm_hwacha_vf_scopy
bli_packm_hwacha_vf_scopy:
	OP vlstw vv0,, va1,, va2
	OP vsw vv0,, va0,
	vstop

	.align 3
	.global bli_packm_hwacha_vf_scopyconvert
bli_packm_hwacha_vf_scopyconvert:
	OP vlstw vv0,, va1,, va2
	OP vfcvt.h.s vv2,, vv0,,
	OP vsh vv2,, va0,,
	vstop

	.align 3
	.global bli_packm_hwacha_vf_sscal2
bli_packm_hwacha_vf_sscal2:
	OP vlstw vv0,, va1,, va2
	OP vfmul.s vv1,, vv0,, vs1
	OP vsw vv1,, va0,,
	vstop

	.align 3
	.global bli_packm_hwacha_vf_sscal2convert
bli_packm_hwacha_vf_sscal2convert:
	OP vlstw vv0,, va1,, va2
	OP vfmul.s vv1,, vv0,, vs1
	OP vfcvt.h.s vv2,, vv1,,
	OP vsh vv2,, va0,,
	vstop
