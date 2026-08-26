# Attribution and third-party notices

## Original pyWannier90

This project is an adaptation of **pyWannier90**, originally written by
**Hung Q. Pham** (`pqh3.14@gmail.com`):

<https://github.com/hungpham2017/pyWannier90>

Original copyright: **Copyright (c) 2017, Hung Pham**.

pyWannier90 is distributed under the BSD 3-Clause License. The original
copyright notice, license conditions, and disclaimer are retained in
[`LICENSE`](LICENSE). The Wannier90-v4 adapter also identifies the original
project and author in its source header.

## Wannier90

Wannier90 is a separate work developed by the Wannier90 Developer Group and
contributors. It is loaded dynamically at runtime and is not included in this
Python distribution. Wannier90 v4 is licensed under LGPL-2.1-or-later:

<https://github.com/wannier-developers/wannier90>

The official repository provides the complete author and contributor list.
The additional authors credited there for the v4 library/interface work are
Jerome Jackson, Leon Petit, and Barry G. Searle.

## PySCF

PySCF is a separate dependency developed by the PySCF authors and contributors
and licensed under Apache-2.0:

<https://github.com/pyscf/pyscf>

## IWF preview implementation

The compatibility investigation considered the independently published IWF
preview interface by Shuoxue Li and Zhi-Hao Cui:

<https://github.com/LiShuoxue/iwf-preview>

No source code from that interface was copied into `pywannier90_v4.py`; the v4
adapter was implemented from the original BSD pyWannier90 code and the official
Wannier90 v4 C/library API. The IWF preview repository is MIT-licensed and its
modified interface itself retains the original pyWannier90 BSD notice.
