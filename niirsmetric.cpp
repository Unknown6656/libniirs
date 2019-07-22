#pragma once

#include "niirsmetric.h"


using namespace std;


const Mat hDiff = (Mat_<char>(1, 2) << 1, -1);
const Mat vDiff = (Mat_<char>(2, 1) << -1, 1);
const Mat hSobel = (Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
const Mat vSobel = (Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);


future<void> NIIRSMetricBase::async_filter2D(const Mat& src, Mat* dst, const int depth, const Mat& filter, const Point& anchor)
{
    return async([&]()
    {
        *dst = Mat(src.rows, src.cols, depth);

        filter2D(src, *dst, src.depth(), filter, anchor);
    });
}

constexpr const double NIIRSMetricBase::normalize(const double value, const double min_bound, const double max_bound)
{
    return 3.0 * (value - max_bound) / (max_bound - min_bound);
}

const double NIIRSMetricBase::RER_BM(const Mat& frame)
{
    Mat hBlur(frame.rows, frame.cols, CV_8UC1);
    Mat vBlur(frame.rows, frame.cols, CV_8UC1);

    blur(frame, hBlur, Size(9, 1));
    blur(frame, vBlur, Size(1, 9));

    Mat dFVer, dFHor, dBVer, dBHor;
    future<void> await0 = async_filter2D(frame, &dFVer, CV_8UC1, vDiff, Point(0, 1));
    future<void> await1 = async_filter2D(frame, &dFHor, CV_8UC1, hDiff, Point(1, 0));
    future<void> await2 = async_filter2D(vBlur, &dBVer, CV_8UC1, vDiff, Point(0, 1));
    future<void> await3 = async_filter2D(hBlur, &dBHor, CV_8UC1, hDiff, Point(1, 0));

    await0.get();
    await1.get();
    await2.get();
    await3.get();

    const Mat dVVer = dFVer - dBVer;
    const Mat dVHor = dFHor - dBHor;
    Mat nVVer(frame.rows, frame.cols, CV_8UC1);

    cv::normalize(dVVer, nVVer, 0, 255, NORM_MINMAX);

    future<double> await4 = async([&]() { return cv::sum(dFVer)[0]; });
    future<double> await5 = async([&]() { return cv::sum(dFHor)[0]; });
    future<double> await6 = async([&]() { return cv::sum(dVVer)[0]; });
    future<double> await7 = async([&]() { return cv::sum(dVHor)[0]; });

    const double sFVer = await4.get();
    const double sFHor = await5.get();
    const double sVVer = await6.get();
    const double sVHor = await7.get();
    const double bFVer = (sFVer == 0) ? 0 : (sFVer - sVVer) / sFVer;
    const double bFHor = (sFHor == 0) ? 0 : (sFHor - sVHor) / sFHor;

    const double BM = bFVer > bFHor ? bFVer : bFHor;

    return 1.17 - 1.15 * BM;
}

const double NIIRSMetricBase::RER_EI(const Mat& frame)
{
    Mat fFrame, dVer, dHor;

    frame.convertTo(fFrame, CV_32F);

    future<void> await0 = async_filter2D(fFrame, &dVer, CV_32F, vSobel, Point(1, 1));
    future<void> await1 = async_filter2D(fFrame, &dHor, CV_32F, hSobel, Point(1, 1));

    await0.get();
    await1.get();

    dVer = dVer.mul(dVer);
    dHor = dHor.mul(dHor);
    dVer = dVer + dHor;

    future<void> await2 = std::async([&]() { cv::sqrt(dVer, dHor); });

    await2.get();

    const double sum = cv::sum(dHor)[0];
    const double EI = (1.0 / frame.rows / frame.cols) * sum;

    return -0.28 + 1.3 * pow(EI * 0.01, 0.25);
}

const double NIIRSMetricBase::RER_FR(const Mat& I)
{
    int windowsize = 1;

    while (windowsize < I.rows && windowsize < I.cols)
        windowsize <<= 1;

    windowsize >>= 1;

    const int windowradius = windowsize / 2;
    const int x = I.cols / 2;
    const int y = I.rows / 2;
    const int lowPassSize = (int)(windowsize * 0.15);
    const Mat padded = I(Rect(x - windowradius, y - windowradius, windowsize, windowsize)); // expand input image to optimal size
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;

    merge(planes, 2, complexI); // Add to the expanded another plane with zeros
    dft(complexI, complexI); // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude

    const Mat norm2I = planes[0].mul(planes[0]);
    const int REG_COUNT = 3;
    const Rect regions[REG_COUNT] =
    {
        Rect(0, 0, lowPassSize, lowPassSize), // Top-Left - Create a ROI per quadrant
        Rect(windowsize - lowPassSize, 0, lowPassSize, lowPassSize), // Top-Right
        Rect(0, windowsize - lowPassSize, lowPassSize, lowPassSize), // Top-Left - Create a ROI per quadrant
//      Rect(windowsize - lowPassSize, windowsize - lowPassSize, lowPassSize, lowPassSize), // Top-Right
    };
    future<double> q[REG_COUNT];

    for (int i = 0; i < REG_COUNT; ++i)
        q[i] = async([&]()
        {
            return cv::sum(norm2I(regions[i]))[0];
        });

    const double q0 = q[0].get();
    const double q1 = q[1].get();
    const double q2 = q[2].get();
    //  const double q3 = q[3].get();
    const double lowPass = q0 + q1 + q2; // + q3;
    const double allPass = cv::sum(norm2I)[0];
    const double highPass = allPass - lowPass;
    const double fr = (lowPass == 0) ? 0 : highPass / lowPass;

    return -0.26 + 3 * pow(fr, 0.25);
}
