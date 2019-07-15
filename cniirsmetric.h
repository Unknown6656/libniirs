#pragma once

#ifndef CNIIRSMETRIC_H
#define CNIIRSMETRIC_H

#include "opencv2/imgproc.hpp"


using namespace cv;

class CNiirsMetric
{
public:
    CNiirsMetric()
    {
    }

    /**
     * Calculates an estimate for the instantaneous image quality. This calculation is independent from FOV and essentially provides an estimate of image degradation in respect to the true scene.
     * 
     * @param colorFrame: A cv::Mat representing the frame to analyze. Runs cv::BGRToGray internally.
     * @return The quality estimate. Range: 0 (best) to -3 (worst).
     */
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
private:
    const double RER_BM(const Mat&);
    const double RER_EI(const Mat&);
    const double RER_FR(const Mat&);
    const double FR(const Mat&);
    constexpr const double normalize(const double, const double, const double);
    constexpr const double clamp(const double, const double, const double);
};

#endif // CNIIRSMETRIC_H
