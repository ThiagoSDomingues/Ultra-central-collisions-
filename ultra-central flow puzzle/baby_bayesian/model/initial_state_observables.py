#!/usr/bin/env python3
"""
initial_state_observables.py

Compute eccentricity cumulants and ratios from TRENTo event outputs.
References:
  - ATLAS Collaboration, JHEP 01 (2020) 51, arXiv:1904.04808 [nucl-ex]
"""

import numpy as np

def eccentricity_2(eps):
    """Second-order eccentricity ε{2} = sqrt(⟨ε²⟩)."""
    m2 = np.mean(eps**2)
    return np.sqrt(m2)

def normalized_4th_cumulant(eps):
    """
    Normalised 4th cumulant: nc₄ = ⟨ε⁴⟩/⟨ε²⟩² - 2.
    nC_n{4, eps} = <ε_n⁴> / <ε_n²>² − 2
    
    Returns: (nCn{4,eps})
    """
    m2 = np.mean(eps ** 2)
    m4 = np.mean(eps ** 4)
    return m4 / m2**2 - 2.0

def ratio_e4_e2(norm_4th_cumulant):
    """Cumulant ratio ε{4}/ε{2}."""
    
    if norm_4th_cumulant > 0:
        ratio = -(norm_4th_cumulant ** 0.25)
    else:
        ratio = (-norm_4th_cumulant) ** 0.25  # norm_cumulant =< 0
    return ratio  

### Add other observables from 4.3 Normalized cumulants and cumulant ratios from arXiv:1904.04808 [nucl-ex]
