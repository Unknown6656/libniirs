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
    virtual const double calculate_absolute(const Mat&, const double, const double) = 0;

protected:
    static const double RER_BM(const Mat&);
    static const double RER_EI(const Mat&);
    static const double RER_FR(const Mat&);

    static const double calculate_gsd(const Mat& colorFrame, const double fov_horizontal, const double fov_vertical)
    {
        const int res_horizontal = colorFrame.cols;
        const int res_vertical = colorFrame.rows;

        // Use larger GSD to anticipate shallow viewing angles
        return (fov_horizontal > fov_vertical) ? fov_horizontal / res_horizontal : fov_vertical / res_vertical;
    }

    static inline std::future<void> async_filter2D(const Mat&, Mat*, const int, const Mat&, const Point&);
    static inline constexpr const double normalize(const double, const double, const double);
    static inline constexpr const double clamp(const double value, const double min, const double max)
    {
        return value < min ? min : value > max ? max : value;
    }
};

#endif
