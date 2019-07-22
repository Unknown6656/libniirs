#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "cvniirsmetric.h"


const double CVNIIRSMetric::calculate_absolute(const Mat& colorFrame, const double fov_horizontal, const double fov_vertical)
{
    // see: https://www.researchgate.net/publication/300792212_Application_of_VNIIRS_for_target_tracking
    // and: 

    const double H = 1; // geometric mean height due to overshoot
    const double Q = 1.618033988749894848; // quality factor in the range of [1..2]
    const double SNR = 40 / 26;
    const double GSD = calculate_gsd(colorFrame, fov_horizontal, fov_vertical);
    const double RER = (RER_BM(colorFrame) + RER_EI(colorFrame) + RER_FR(colorFrame) + 0.3) / 4.0;

    const double a = RER < 0.9 ? 3.16 : 3.32;
    const double b = RER < 0.9 ? 2.817 : 1.559;
    const double M = RER < 0.9 ? 11.53 : 11.6;
    const double NIIRS_EO = 10.251 - a * log10(GSD) + b * log10(RER) - 0.656 * H - 0.344 * SNR;
    const double NIIRS_SAR = 10.751 - a * log10(GSD) + b * log10(RER) - 0.656 * H - 0.344 * SNR;
    const double NIIRS_IR = 1.14 + 0.18 * NIIRS_SAR + 0.08 * NIIRS_SAR * NIIRS_SAR;
    const double GSD_SAR = pow(10, (10.751 - NIIRS_IR) / a);

    const double VNIIRS_MIQE = M - a * log10(GSD) + 2 * log10(Q) + b * log10(RER) - 0.656 * H - 0.344 * SNR;

    return (VNIIRS_MIQE + NIIRS_EO + NIIRS_IR + NIIRS_SAR) / 4;
}
