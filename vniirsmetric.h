#pragma once

#ifndef VNIIRSMETRIC_H
#define VNIIRSMETRIC_H

#include "niirsmetric.h"


class VNIIRSMetric : public NIIRSMetricBase
{
public:
    VNIIRSMetric()
        : NIIRSMetricBase()
    {
    }

    const double calculate(const Mat& colorFrame);
};

#endif
