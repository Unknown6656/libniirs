#pragma once

#ifndef VNIIRSMETRIC_H
#define VNIIRSMETRIC_H

#include "niirsmetric.h"


class VNiirsMetric : public MetricBase
{
public:
    VNiirsMetric()
        : MetricBase()
    {
    }

    const double calculate(const Mat& colorFrame);
};

#endif
