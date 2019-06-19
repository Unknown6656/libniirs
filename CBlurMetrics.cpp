#include "CBlurMetrics.h"


#include <iostream>

// OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory>
#include <fstream>
#include <string>

CBlurMetrics::CBlurMetrics(){
}


double CBlurMetrics::calculate(const cv::Mat& imgGray){
    cv::Mat doubleImg;
    imgGray.convertTo(doubleImg,CV_64F);

    cv::Scalar stdDev, mean;
    double focusMeasure; // = 0.0: (performance) Variable 'focusMeasure' is reassigned a value before the old one has been used.
    focusMeasure = computeNormalizedVarianceOfGradient(doubleImg,mean,stdDev);
    //std::cout << "Mean of Gradient = " <<  mean.val[0] << " StdDev of Gradient = " << stdDev.val[0] << " normalized=" << focusMeasure << std::endl;

    /*
       focusMeasure = normalizedGraylevelVariance(doubleImg);
       std::cout << "focusMeasureNormalizedGraylevelVariance = " << focusMeasure << std::endl;


       focusMeasure = varianceOfLaplacian(doubleImg);
       std::cout << "focusMeasureLaplacian = " << focusMeasure << std::endl;

       focusMeasure = tenengrad(doubleImg,3);
       std::cout << "focusMeasureTenengrad = " << focusMeasure << std::endl;

       focusMeasure = modifiedLaplacian(doubleImg);
       std::cout << "focusMeasureModifiedLaplacian = " << focusMeasure << std::endl;

       double cannyThresh1=175, cannyThresh2=225;
       CCpbd* cpbd_obj = new CCpbd();
       cpbd_obj->findLowAndHighImageThresholds(imgGray,cannyThresh1,cannyThresh2);
       std::cout << "cannyThresh1=" << cannyThresh1 << " cannyThresh2=" << cannyThresh2 << std::endl;
       delete cpbd_obj;

       focusMeasure = computeBlurrinesWithCanny(imgGray,cannyThresh1,cannyThresh2);
       std::cout << "focusMeasureCanny = " << focusMeasure << std::endl;
     */
    return focusMeasure;
}

double CBlurMetrics::calculate(const cv::Mat& imgGray, std::string fname){
    cv::Mat doubleImg;
    imgGray.convertTo(doubleImg,CV_64F);

    cv::Scalar stdDev, mean;
    double focusMeasure; // = 0.0: (performance) Variable 'focusMeasure' is reassigned a value before the old one has been used.
    focusMeasure = computeNormalizedVarianceOfGradient(doubleImg,mean,stdDev, fname);
    return focusMeasure;
}



double CBlurMetrics::modifiedLaplacian(const cv::Mat& doubleImg)
{
    cv::Mat M = (cv::Mat_<double>(3, 1) << -1, 2, -1);
    cv::Mat G = cv::getGaussianKernel(3, -1, CV_64F);

    cv::Mat Lx;
    cv::sepFilter2D(doubleImg, Lx, CV_64F, M, G);

    cv::Mat Ly;
    cv::sepFilter2D(doubleImg, Ly, CV_64F, G, M);

    cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

    double focusMeasure = cv::mean(FM).val[0];
    return focusMeasure;
}

// OpenCV port of 'LAPV' algorithm (Pech2000)
double CBlurMetrics::varianceOfLaplacian(const cv::Mat& doubleImg)
{
    cv::Mat lap;
    cv::Laplacian(doubleImg, lap, CV_64F);

    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);

    double focusMeasure = sigma.val[0]*sigma.val[0];
    return focusMeasure;
}

// OpenCV port of 'TENG' algorithm (Krotkov86)
double CBlurMetrics::tenengrad(const cv::Mat& doubleImg, int ksize)
{
    cv::Mat Gx, Gy;
    cv::Sobel(doubleImg, Gx, CV_64F, 1, 0, ksize);
    cv::Sobel(doubleImg, Gy, CV_64F, 0, 1, ksize);

    cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);

    double focusMeasure = sqrt(cv::mean(FM).val[0]);
    return focusMeasure;
}

// OpenCV port of 'GLVN' algorithm (Santos97)
double CBlurMetrics::normalizedGraylevelVariance(const cv::Mat& doubleImg)
{
    cv::Scalar mu, sigma;
    cv::meanStdDev(doubleImg, mu, sigma);

    double focusMeasure = (sigma.val[0]*sigma.val[0]) / mu.val[0];

    //std::cout << "NormalizedGrayLevelVariance: standardDev=" << sigma.val[0] << " mean=" << mu.val[0] << std::endl;
    return focusMeasure;
}


double CBlurMetrics::computeVariance(cv::Mat& doubleImg, cv::Scalar& meanVal, cv::Scalar& varianceVal)
{
    meanVal = cv::mean(doubleImg);
    cv::Mat outImg;
    cv::subtract(doubleImg,meanVal,outImg);
    varianceVal = cv::sum(outImg.mul(outImg));
    varianceVal /= (doubleImg.rows * doubleImg.cols);
    return varianceVal.val[0];
}

double CBlurMetrics::computeNormalizedVarianceOfGradient(const cv::Mat& doubleImg, cv::Scalar& mean, cv::Scalar& stdDev)
{
    cv::Mat sobelGradientY, sobelGradientX;
    cv::Sobel(doubleImg, sobelGradientX,CV_16S, 1, 0);
    cv::Sobel(doubleImg, sobelGradientY, CV_16S, 0, 1);
    //cv::Mat absGradX, absGradY;
    //convertScaleAbs(sobelGradientX,absGradX);
    //convertScaleAbs(sobelGradientY,absGradY);
    cv::Mat gradMag;
    //cv::add(absGradX,absGradY,gradMag);
    cv::add(abs(sobelGradientX),abs(sobelGradientY),gradMag);

    cv::meanStdDev(gradMag,mean,stdDev);
    return (stdDev.val[0] * stdDev.val[0]) / mean.val[0];
}

double CBlurMetrics::computeNormalizedVarianceOfGradient(const cv::Mat& doubleImg, cv::Scalar& mean, cv::Scalar& stdDev, std::string fname)
{
    cv::Mat sobelGradientY, sobelGradientX;
    cv::Sobel(doubleImg, sobelGradientX, CV_16S, 1, 0);
    cv::Sobel(doubleImg, sobelGradientY, CV_16S, 0, 1);
    //cv::Mat absGradX, absGradY;
    //convertScaleAbs(sobelGradientX,absGradX);
    //convertScaleAbs(sobelGradientY,absGradY);
    cv::Mat gradMag;
    cv::Mat argh;
    //cv::add(absGradX,absGradY,gradMag);
    cv::add(abs(sobelGradientX),abs(sobelGradientY),gradMag);

    gradMag.convertTo(argh, CV_8U);

    int histSize = 256;
    float range[] = { 0, 255 };
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    cv::Mat b_hist;
    cv::calcHist(&argh, 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::normalize(b_hist, b_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    std::ofstream histOutFile = std::ofstream(fname);
    for(int i = 0; i < 256; i+=2) {
        histOutFile << b_hist.at<float>(i) << '\n';
    }
    histOutFile.flush();
    histOutFile.close();

    cv::meanStdDev(gradMag,mean,stdDev);
    return (stdDev.val[0] * stdDev.val[0]) / mean.val[0];
}

double CBlurMetrics::computeBlurrinesWithCanny(const cv::Mat& grayImg, double cannyThresh1, double cannyThresh2)
{
    cv::Mat cannyEdge;
    cv::Canny(grayImg, cannyEdge, cannyThresh1, cannyThresh2);
    //imwrite("Canny1.jpg",cannyEdge);
    int noEdgePixels = cv::countNonZero(cannyEdge);
    //std::cout << "noEdgePixels = " << noEdgePixels << std::endl;
    double sharpness = (noEdgePixels * 100.0) / (cannyEdge.rows * cannyEdge.cols);
    return sharpness;
}

static int min_dct_value = 1;   /* -d= */
static double max_histogram_value = 0.005; /* -h= */

static double weights[] = {     /* diagonal weighting */
    8,7,6,5,4,3,2,1,
    1,8,7,6,5,4,3,2,
    2,1,8,7,6,5,4,3,
    3,2,1,8,7,6,5,4,
    4,3,2,1,8,7,6,5,
    5,4,3,2,1,8,7,6,
    6,5,4,3,2,1,8,7,
    7,6,5,4,3,2,1,8
};
static double total_weight = 344;
