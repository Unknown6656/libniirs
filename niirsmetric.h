#pragma once

#ifndef METRIC_H
#define METRIC_H

#include <future>
#include <math.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


using namespace cv;


class NIIRSMetricBase
{
public:
    /**
     * Calculates an estimate for the instantaneous image quality. This calculation is independent from FOV and essentially provides an estimate of image degradation in respect to the true scene.
     *
     * @param colorFrame: A cv::Mat representing the frame to analyze. Runs cv::BGRToGray internally.
     * @return The quality estimate. Range: 0 (best) to -3 (worst).
     */
    virtual const double calculate(const Mat& colorFrame) = 0;

protected:
    static const double RER_BM(const Mat&);
    static const double RER_EI(const Mat&);
    static const double RER_FR(const Mat&);

    static inline std::future<void> async_filter2D(const Mat&, Mat*, const int, const Mat&, const Point&);
    static inline constexpr const double normalize(const double, const double, const double);
    static inline constexpr const double clamp(const double value, const double min, const double max)
    {
        return value < min ? min : value > max ? max : value;
    }
};

#endif
