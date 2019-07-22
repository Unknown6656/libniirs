#pragma once

#ifndef CNIIRSMETRIC_H
#define CNIIRSMETRIC_H

#include "niirsmetric.h"


class CNiirsMetric : public NIIRSMetricBase
{
public:
    CNiirsMetric()
        : NIIRSMetricBase()
    {
    }

    const double calculate(const Mat& colorFrame);

    /**
     * Estimates instantaneous Video-NIIRS level of the frame provided. Uses information about the current field-of-view of the observer to calculate the absolute Ground Resolvable Distance (GRD) and returns the associated (logarithmic) NIIRS level.
     *
     * @param colorFrame: A cv::Mat representing the frame to analyze. Runs cv::BGRToGray internally.
     * @param fov_horizontal: The width of the area covered by the frame in meters.
     * @param fov_vertical: The height of the area covered by the frame in meters.
     * @return The NIIRS estimate. Range: 2 (worst) to 11 (best).
     */
    const double calculate_absolute(const Mat& colorFrame, const double fov_horizontal, const double fov_vertical);
    const double calculate_absolute(const Mat& colorFrame, const double pniirs_theoretical);
};

#endif
