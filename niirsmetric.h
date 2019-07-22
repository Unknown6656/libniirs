#pragma once

#ifndef METRIC_H
#define METRIC_H

#include <future>
#include <math.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


using namespace cv;

class MetricBase
{
public:
    /**
     * Calculates an estimate for the instantaneous image quality. This calculation is independent from FOV and essentially provides an estimate of image degradation in respect to the true scene.
     *
     * @param colorFrame: A cv::Mat representing the frame to analyze. Runs cv::BGRToGray internally.
     * @return The quality estimate. Range: 0 (best) to -3 (worst).
     */
    virtual const double calculate(const Mat& colorFrame) = 0;
};

#endif
