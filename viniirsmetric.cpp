#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "vniirsmetric.h"


const double VNIIRSMetric::calculate(const Mat& colorFrame)
{
    double GSD; // ground sampling distance
    double RER; // relative edge response
    double SNR; // signal noise ratio
    double G; // noise gain
    double H; // geometric mean height due to overshoot

    double Q; // quality factor in the range of [1..2]

    const double a = RER < 0.9 ? 3.16 : 3.32;
    const double b = RER < 0.9 ? 2.817 : 1.559;
    const double M = RER < 0.9 ? 11.53 : 11.6;
    const double NIIRS_EO = 10.251 - a * log10(GSD) + b * log10(RER) - 0.656 * H - 0.344 * G / SNR;
    const double NIIRS_SAR = 10.751 - a * log10(GSD) + b * log10(RER) - 0.656 * H - 0.344 * G / SNR;
    const double NIIRS_IR = 1.14 + 0.18 * NIIRS_SAR + 0.08 * NIIRS_SAR * NIIRS_SAR;
    const double GSD_SAR = pow(10, (10.751 - NIIRS_IR) / a);

    const double VNIIRS_MIQE = M - a * log10(GSD) + 2 * log10(Q) + b * log10(RER) - 0.656 * H - 0.344 * G / SNR;


    // TODO

    return 0;
}
