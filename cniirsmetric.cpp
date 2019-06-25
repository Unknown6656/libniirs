#include "cniirsmetric.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

//
#define NIIRS_MIN 3.004
#define NIIRS_MAX 4.48925
#define EIR_MIN 3.18458
#define EIR_MAX 14.9964
#define FR_MIN 1.31376e-005
#define FR_MAX 0.0056512

using namespace cv;
using namespace std;

const Mat hDiff = (Mat_<char>(1, 2) << 1, -1);
const Mat vDiff = (Mat_<char>(2, 1) << -1, 1);
const Mat hSobel = (Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
const Mat vSobel = (Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

CNiirsMetric::CNiirsMetric()
{
}

int *testBlocks;
Mat src; Mat src_gray;


double CNiirsMetric::RER_BM(Mat &frame) {
    Mat hBlur(frame.rows, frame.cols, CV_8UC1);
    Mat vBlur(frame.rows, frame.cols, CV_8UC1);

    blur(frame, hBlur, Size(9, 1));
    blur(frame, vBlur, Size(1, 9));

    Mat dFVer(frame.rows, frame.cols, CV_8UC1);
    Mat dFHor(frame.rows, frame.cols, CV_8UC1);
    Mat dBVer(frame.rows, frame.cols, CV_8UC1);
    Mat dBHor(frame.rows, frame.cols, CV_8UC1);
    filter2D(frame, dFVer, frame.depth(), vDiff, Point(0, 1));
    filter2D(frame, dFHor, frame.depth(), hDiff, Point(1, 0));
    filter2D(vBlur, dBVer, vBlur.depth(), vDiff, Point(0, 1));
    filter2D(hBlur, dBHor, hBlur.depth(), hDiff, Point(1, 0));

    Mat dVVer = dFVer - dBVer;
    Mat dVHor = dFHor - dBHor;

    Mat nVVer(frame.rows, frame.cols, CV_8UC1);
    cv::normalize(dVVer, nVVer, 0, 255, NORM_MINMAX);

    double vMax, hMax;
    minMaxLoc(dVVer, NULL, &vMax);
    minMaxLoc(dVHor, NULL, &hMax);

    double sFVer = cv::sum(dFVer)[0];
    double sFHor = cv::sum(dFHor)[0];
    double sVVer = cv::sum(dVVer)[0];
    double sVHor = cv::sum(dVHor)[0];

    double bFVer = (sFVer == 0) ? 0 : (sFVer - sVVer) / sFVer;
    double bFHor = (sFHor == 0) ? 0 : (sFHor - sVHor) / sFHor;

    double BM = bFVer > bFHor ? bFVer : bFHor;
    return 1.17 - 1.15 * BM;
}


double CNiirsMetric::RER_EI(Mat &frame) {
    Mat fFrame;
    frame.convertTo(fFrame, CV_32F);
    Mat dVer(frame.rows, frame.cols, CV_32F);
    Mat dHor(frame.rows, frame.cols, CV_32F);
    filter2D(fFrame, dVer, fFrame.depth(), vSobel, Point(1, 1));
    filter2D(fFrame, dHor, fFrame.depth(), hSobel, Point(1, 1));

    dVer = dVer.mul(dVer);
    dHor = dHor.mul(dHor);
    dVer = dVer + dHor;

    cv::sqrt(dVer, dHor);
    Mat rFrame;
    dHor.convertTo(rFrame, CV_8UC1, 1, 128);

    double sum = cv::sum(dHor)[0];
    double EI = (1.0 / frame.rows / frame.cols) * sum;
    return -0.28 + 1.3*pow(EI * 0.01, 0.25);
}

double CNiirsMetric::RER_FR(Mat &I) {
    int windowsize = 1;
    while (windowsize < I.rows && windowsize < I.cols) {
        windowsize <<= 1;
    }
    windowsize >>= 1;

    int windowradius = windowsize / 2;
    int x = I.cols / 2;
    int y = I.rows / 2;
    int lowPassSize = (int) (windowsize * 0.15);

    Mat padded = I(Rect(x - windowradius, y - windowradius, windowsize, windowsize)); //expand input image to optimal size

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI); // Add to the expanded another plane with zeros

    dft(complexI, complexI); // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude

    Mat magI = planes[0];
    Mat normSquareI = magI.mul(magI);

    double min, max;
    minMaxLoc(normSquareI, &min, &max);

    //qDebug() << "normSquareI range: " << min << " " << max << "\n";

    Mat q0(normSquareI, Rect(0, 0, lowPassSize, lowPassSize)); // Top-Left - Create a ROI per quadrant
    Mat q1(normSquareI, Rect(windowsize - lowPassSize, 0, lowPassSize, lowPassSize)); // Top-Right
    Mat q2(normSquareI, Rect(0, windowsize - lowPassSize, lowPassSize, lowPassSize)); // Top-Left - Create a ROI per quadrant
    Mat q3(normSquareI, Rect(windowsize - lowPassSize, windowsize - lowPassSize, lowPassSize, lowPassSize)); // Top-Right


    double lowPass = cv::sum(q0)[0]
                     + cv::sum(q1)[0]
                     + cv::sum(q2)[0];
    //+ cv::sum(q3)[0];
    double allPass = cv::sum(normSquareI)[0];
    double highPass = allPass - lowPass;
    double fr = (lowPass == 0) ? 0 : highPass / lowPass;
    return -0.26 + 3 * pow(fr, 0.25);
}

double percentile(Mat frame, double percentile_number) {
    Mat src_gray, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Load an image

    if (!frame.data)
    {
        return -1;
    }

    /// Remove noise by blurring with a Gaussian filter
    GaussianBlur(frame, frame, Size(3, 3), 0, 0, BORDER_DEFAULT);
    /// Convert the image to grayscale
    src_gray = frame;

    /// Apply Laplace function
    Mat abs_dst, norm_abs_dst;

    Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(dst, abs_dst);
    normalize(abs_dst, norm_abs_dst, 0, 200, NORM_MINMAX);


    /// Establish the number of bins
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 255 };
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist;

    /// Compute the histograms:
    calcHist(&abs_dst, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

    double totalCount = 0;
    double mean = 0;
    for (int i = 0; i < 256; i++) {
        totalCount += b_hist.at<float>(i);
        mean += i * b_hist.at<float>(i);
    }
    mean /= totalCount;


    double percCount = totalCount;
    int percentile_position = 255;
    for (; percentile_position >= 0; percentile_position--) {
        percCount -= b_hist.at<float>(percentile_position);
        if (percCount / totalCount < percentile_number) {
            break;
        }
    }

    //cout << totalCount << "/" << mean << "/" << percentile_position << endl;
    return percentile_position / mean;
}

double CNiirsMetric::normalize(double value, double min_bound, double max_bound) {
    double mapped = 3.0 * (value - max_bound) / (max_bound - min_bound);
    return mapped;
}

double CNiirsMetric::calculate(Mat &colorFrame) {
    // MISB metrics use grayscale

    Mat frame;
    cv::cvtColor(colorFrame, frame, CV_BGR2GRAY);

    // Blur metrics according to MISB
    double rer_bm = RER_BM(frame);
    double rer_ei = RER_EI(frame);
    double rer_fr = RER_FR(frame);

    // Average metric outputs
    double rer = (rer_bm + rer_ei + rer_fr + 0.3) / 4.0;
    double blurriness = -log2(rer);

    // Clamp result to  [0..3]
    if(blurriness < 0) blurriness = 0;
    if(blurriness > 3) blurriness = 3;
    return blurriness;
}


double CNiirsMetric::calculate_absolute(Mat &colorFrame, double fov_horizontal, double fov_vertical) {
    int res_horizontal = colorFrame.cols;
    int res_vertical = colorFrame.rows;
    // Use larger GSD to anticipate shallow viewing angles
    double gsd_max = (fov_horizontal > fov_vertical) ? fov_horizontal / res_horizontal : fov_vertical / res_vertical;

    // Calculate MISB-like blur metrics
    double blurriness = calculate(colorFrame); // [0..3]

    // Bring into absulte form
    double niirs_misb = 5.0 - log2(gsd_max) - blurriness;
    return niirs_misb;
}



double CNiirsMetric::calculate_absolute(Mat &colorFrame, double pniirs_theoretical) {
    double blurriness = calculate(colorFrame); // [0..3]
    // Bring into absulte form
    double niirs_misb = pniirs_theoretical - blurriness;
    return niirs_misb;
}
