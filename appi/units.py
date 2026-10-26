"""Affinity conventions: stronger binding means larger pKd and negative ddG."""

import math

R_KCAL_MOL_K = 0.00198720425864083
DEFAULT_TEMPERATURE_K = 298.15


def energy_factor(temperature_k=DEFAULT_TEMPERATURE_K):
    if not math.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Temperature must be finite and positive (kelvin)")
    return R_KCAL_MOL_K * temperature_k * math.log(10)


def pkd_from_molar(kd_molar):
    if not math.isfinite(kd_molar) or kd_molar <= 0:
        raise ValueError("Kd must be finite and positive (molar)")
    return -math.log10(kd_molar)


def delta_pkd_to_ddg(delta_pkd, temperature_k=DEFAULT_TEMPERATURE_K):
    return -energy_factor(temperature_k) * delta_pkd


def ddg_to_delta_pkd(ddg, temperature_k=DEFAULT_TEMPERATURE_K):
    return -ddg / energy_factor(temperature_k)

