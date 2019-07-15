#pragma once

#ifndef CBLURMETRICS_H
#define CBLURMETRICS_H

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


/**
 * @brief The CBlurMetrics class
 * implements various metrics to measure
 * the absolute bluriness of a grayscale image
 */
class CBlurMetrics
{
public:
    CBlurMetrics()
    {
    }

    ~CBlurMetrics()
    {
    }

    /**
     * Computes the blurriness of a grayscale image
     * by calling one of the computeX method
     * the best metric seems to be computeNormalizedVarianceOfGradient()
     * @brief calculate
     * @param img
     * @return the blurriness metric
     */
    double calculate(const cv::Mat& img);

    /**
     * Computes the blurriness of a grayscale image
     * by calling one of the computeX method
     * the best metric seems to be computeNormalizedVarianceOfGradient()
     * @brief calculate
     * @param img
     * @return the blurriness metric
     */
    double calculate(const cv::Mat& img, std::string fname);

    /**
     * Computes the mean and stdDev of the grayscale image doubleImg
     * and returns the normalized variance
     * @brief normalizedGraylevelVariance
     * @param doubleImg - the input image as CV_64F
     * @return (stdDev*stdDev) / mean
     */
    double normalizedGraylevelVariance(const cv::Mat& doubleImg);

    /**
     * Computes the mean and variance of the grayscale image
     * and returns the variance
     * @brief computeVariance
     * @param doubleImg - input image as CV_64F
     * @param meanVal
     * @param varianceVal
     * @return varianceVal as double
     */
    double computeVariance(cv::Mat& doubleImg, cv::Scalar& meanVal, cv::Scalar& varianceVal);

    /**
     * LAPM algorithm Nayar89
     * Computes the gradient by a linear filter with the kernel (-1,2,-1)
     * it returns the mean of the gradient image
     * @brief modifiedLaplacian
     * @param doubleImg - input image as CV_64F
     * @return
     */
    double modifiedLaplacian(const cv::Mat& doubleImg);

    /**
     * LAPV algorithm Pech2000
     * Computes the variance of the Laplacian image
     * @brief varianceOfLaplacian
     * @param doubleImg - input image as CV_64F
     * @return
     */
    double varianceOfLaplacian(const cv::Mat& doubleImg);

    /**
     * TENG algorithm Krotkov86
     * Computes the magnitude of the Sobel gradient
     * it returns the mean value of it
     * @brief tenengrad
     * @param doubleImg - input image as CV_64F
     * @param ksize - kernel size, default 3
     * @return
     */
    double tenengrad(const cv::Mat& doubleImg, int ksize = 3);

    /**
     * Computes the Sobel gradient image
     * the mean and stdDev of it
     * and returns stdDev*stdDev / mean
     * @brief computeNormalizedVarianceOfGradient
     * @param doubleImg - input image as CV_64F
     * @param mean
     * @param stdDev
     * @return
     */
    double computeNormalizedVarianceOfGradient(const cv::Mat& doubleImg, cv::Scalar& mean, cv::Scalar& stdDev);

    /**
     * Computes the Sobel gradient image
     * the mean and stdDev of it
     * and returns stdDev*stdDev / mean
     * also outputs the historgram into a file
     * @brief computeNormalizedVarianceOfGradient
     * @param doubleImg - input image as CV_64F
     * @param mean
     * @param stdDev
     * @param fname - name of the file to write the histogram into
     * @return
     */
    double computeNormalizedVarianceOfGradient(const cv::Mat& doubleImg, cv::Scalar& mean, cv::Scalar& stdDev, std::string fname);

    /**
     * Computes the canny edge image with the given low and high thresholds,
     * counts the edge pixels in the canny edge image
     * and returns the percent of the edge pixels relativ to the image size
     * @brief computeBlurrinesWithCanny
     * @param grayImg - input grayscale image CV_8U
     * @param cannyThresh1
     * @param cannyThresh2
     * @return
     */
    double computeBlurrinesWithCanny(const cv::Mat& grayImg, double cannyThresh1, double cannyThresh2);

    /**
     * Computes the sharpness via dct coefficients after marichal-ma-zhang
     * https://github.com/tokenrove/blur-detection/blob/master/README.org
     * @brief dct
     * @param path - file name of the input image
     * @return
     */
    double computeDCT(const char* path);
private:
    /**
     * @brief update_histogram - method used in computeDCT
     * @param block
     * @param histogram
     */
    void update_histogram(short* block, int* histogram);

    /**
     * @brief compute_blur - method used in computeDCT
     * @param histogram
     * @return
     */
    double compute_blur(int* histogram);
};

#endif
