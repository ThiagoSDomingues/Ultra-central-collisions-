#!/usr/bin/env python3
"""
Generalized script to calculate initial state observables from TRENTo outputs.
References:
  - ATLAS Collaboration, JHEP 01 (2020) 51, arXiv:1904.04808 [nucl-ex]
"""

import numpy as np

def nCn_4(eps):
    """
    Normalized eccentricities cumulants nCn{4,eps}
    Based on ATLAS definitions (Eq. 8 in arXiv:1904.04808)
    nc_n = <ε_n⁴> / <ε_n²>² − 2
    
    Returns: (nCn{4,eps})
    
    """
    m2 = np.mean(eps ** 2)
    m4 = np.mean(eps ** 4)
    return m4 / m2**2 - 2.0

### cumulant ratio of εn{4} to εn{2}
def ratio_en4_to_en2(norm_cumulant_4):
    """
    """
    if norm_cumulant_4 > 0:
        ratio = -(norm_cumulant_4 ** 0.25)
    else:
        ratio = (-norm_cumulant_4) ** 0.25  # norm_cumulant =< 0
    return ratio  

### Add other observables from 4.3 Normalized cumulants and cumulant ratios from arXiv:1904.04808 [nucl-ex]
