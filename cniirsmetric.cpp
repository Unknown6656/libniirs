#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "cniirsmetric.h"


#define NIIRS_MIN 3.004
#define NIIRS_MAX 4.48925
#define EIR_MIN 3.18458
#define EIR_MAX 14.9964
#define FR_MIN 1.31376e-005
#define FR_MAX 0.0056512

using namespace std;


int* testBlocks;
Mat src, src_gray;


const double CNIIRSMetric::calculate(const Mat& colorFrame)
{
    Mat frame;

    cv::cvtColor(colorFrame, frame, CV_BGR2GRAY);    // MISB metrics use grayscale

    // async 'n' shiet
    auto rer_bm__ = async(&RER_BM, frame);
    auto rer_ei__ = async(&RER_EI, frame);

    // Blur metrics according to MISB
    const double rer_fr = RER_FR(frame);
    const double rer_bm = rer_bm__.get();
    const double rer_ei = rer_ei__.get();

    // Average metric outputs
    const double rer = (rer_bm + rer_ei + rer_fr + 0.3) / 4.0;
    double blurriness = -log2(rer);

    // Clamp result to  [0..3]
    return clamp(blurriness, 0, 3);
}

const double CNIIRSMetric::calculate_absolute(const Mat& colorFrame, const double fov_horizontal, const double fov_vertical)
{
    // Calculate MISB-like blur metrics
    const double blurriness = calculate(colorFrame); // [0..3]
    const double gsd = calculate_gsd(colorFrame, fov_horizontal, fov_vertical);

    // Bring into absulte form
    return 5.0 - log2(gsd) - blurriness;
}

const double CNIIRSMetric::calculate_absolute(const Mat& colorFrame, const double pniirs_theoretical)
{
    const double blurriness = calculate(colorFrame); // [0..3]

    // Bring into absulte form
    return pniirs_theoretical - blurriness;
}
