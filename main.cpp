#include "cniirsmetric.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

/*
 *
 *
 *
 */
int main0(int argc, char** argv) {
    std::cout <<"kek\n";
    CNiirsMetric metric;

    VideoCapture video("C:/users/dennis/desktop/random/iosb/eorakos/MVI_0799_VIS_OB.avi");
    if(!video.isOpened()) {
        std::cout << "(bad video) slice <videofile> <startframe> <endframe> <skip-frames>";
        return 0;
    }

    while(video.grab()) {
        cv::Mat frame;
        video.read(frame);
        double niirs = metric.calculate_absolute(frame, 500, 300);
        std::cout << niirs << "\n";
    }
    return 0;
}



/*
 * int main()
 * for file in D:\cancer:
 *   niirs = metric.calculate_absolute(mat(file), atof(file))
 *   fout << filename << ":" << niirs << "\n";
 *
 *
 *
 *
 * */
