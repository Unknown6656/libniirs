#pragma once

#ifndef CVNIIRSMETRIC_H
#define CVNIIRSMETRIC_H

#include "niirsmetric.h"


class CVNIIRSMetric : public NIIRSMetricBase
{
public:
    CVNIIRSMetric()
        : NIIRSMetricBase()
    {
    }

    const double calculate_absolute(const Mat&, const double, const double);
};

#endif
