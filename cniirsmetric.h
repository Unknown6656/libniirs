#ifndef CNIIRSMETRIC_H
#define CNIIRSMETRIC_H

#include "opencv2/imgproc.hpp"

class CNiirsMetric
{
public:
CNiirsMetric();

/* Calculates an estimate for the instantaneous image quality. This calculation is independent from FOV and essentially provides an estimate of image degradation in respect to the true scene.
 * Parameters:
 * colorFrame: A cv::Mat representing the frame to analyze. Runs cv::BGRToGray internally.
 *
 * Returns:
 * The quality estimate. Range: 0 (best) to -3 (worst).
 */
double calculate(cv::Mat &colorFrame);

/* Estimates instantaneous Video-NIIRS level of the frame provided. Uses information about the current field-of-view of the observer to calculate the absolute Ground Resolvable Distance (GRD) and returns the associated (logarithmic) NIIRS level.
 *
 * Parameters:
 * colorFrame: A cv::Mat representing the frame to analyze. Runs cv::BGRToGray internally.
 * fov_horizontal: The width of the area covered by the frame in meters.
 * fov_vertical: The height of the area covered by the frame in meters.
 *
 * Returns:
 * The NIIRS estimate. Range: 2 (worst) to 11 (best).
 */
double calculate_absolute(cv::Mat &colorFrame, double fov_horizontal, double fov_vertical);


double calculate_absolute(cv::Mat &colorFrame, double pniirs_theoretical);

private:
double RER_BM(cv::Mat&);
double RER_EI(cv::Mat&);
double RER_FR(cv::Mat&);
double FR(cv::Mat&);
double normalize(double, double, double);

};

#endif // CNIIRSMETRIC_H
