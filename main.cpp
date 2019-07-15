#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <regex>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "cniirsmetric.h"


using namespace cv;
namespace fs = std::experimental::filesystem;

#define ENTRY_POINT main_03


/*  Command line:
 *      ./libniirs <inputdir> <outputdir>
 */
const int main_01(const int argc, const char** argv)
{
    if (argc < 3)
    {
        std::cout << "At least two parameters are required (the input and output directories).";

        return -1;
    }

    const std::string dir_in = argv[1];
    const std::string dir_out = argv[2];
    const std::string csv_path = dir_out + "/niirs_levels.csv";
    const std::regex reg_pn("\\$PNIRS,\\d+,([0-9\\.]+),");
    const std::regex reg_wl("\\$PMJC3,([0-9\\.]+),([0-9\\.]+),([0-9\\.]+),([0-9\\.]+),([0-9\\.]+),([0-9\\.]+),([0-9\\.]+),([0-9\\.]+)");
    std::smatch match;
    std::ofstream csv(csv_path);
    CNiirsMetric metric;

    for (const auto& entry : fs::directory_iterator(dir_in))
    {
        const fs::path path = entry.path();
        const std::string opath = dir_out + "/" + path.filename().generic_string() + "--annotated" + path.extension().generic_string();
        Mat frame = imread(path.generic_string());

        std::cout << "Reading \"" << path << "\"..." << std::endl;

        if (frame.empty())
        {
            std::cout << "Unable to read \"" << path << "\"." << std::endl;

            continue;
        }

        std::ifstream img(path);
        std::vector<char> buffer(2048);

        img.read(&buffer[0], buffer.size());

        for (int i = 0, l = buffer.size(); i < l; ++i)
            if (!buffer[i])
                buffer[i] = ' ';

        const std::string binary(buffer.begin(), buffer.end());
        std::regex_search(binary, match, reg_pn);

        img.close();

        const double pniirs = atof(match[1].str().c_str());
        double niirs;

        if ((int)pniirs)
        {
            std::cout << "   P-NIIRS for \"" << path << "\": " << pniirs << std::endl;
            niirs = metric.calculate_absolute(frame, pniirs);
        }
        else
        {
            std::cout << "could not find $PNIRS-metadata entry. Looking for world file instead ..." << std::endl;
            std::string wld_path = dir_in + path.filename().generic_string();

            wld_path = wld_path.replace(wld_path.length() - path.extension().generic_string().length(), path.extension().generic_string().length() + 1, ".wld");

            std::cout << "     opening \"" << wld_path << "\"..." << std::endl;
            std::ifstream wld_fs(wld_path);
            std::string wld((std::istreambuf_iterator<char>(wld_fs)), std::istreambuf_iterator<char>());

            std::regex_search(wld, match, reg_wl);

            const double coord[8] =
            {
                atof(match[1].str().c_str()), // LAT TOP LEFT
                atof(match[2].str().c_str()), // LON TOP LEFT
                atof(match[3].str().c_str()), // LAT TOP RIGHT
                atof(match[4].str().c_str()), // LON TOP RIGHT
                atof(match[5].str().c_str()), // LAT BOTTOM RIGHT
                atof(match[6].str().c_str()), // LON BOTTOM RIGHT
                atof(match[7].str().c_str()), // LAT BOTTOM LEFT
                atof(match[8].str().c_str()), // LON BOTTOM LEFT
            };

            wld_fs.close();

            // 1lat == 111.111km
            // 1lon == 111.111 * cos(1rad lat)
            const double delta_lat1 = std::abs(coord[0] - coord[2]) * 111111;
            const double delta_lat2 = std::abs(coord[6] - coord[4]) * 111111;
            const double delta_lon1 = std::abs(coord[1] - coord[3]) * 111111 * std::cos((coord[0] - coord[2]) * 0.01745329251);
            const double delta_lon2 = std::abs(coord[7] - coord[5]) * 111111 * std::cos((coord[6] - coord[4]) * 0.01745329251);
            const double diag = std::sqrt(std::pow((delta_lat1 + delta_lat2) / 2, 2) + std::pow((delta_lon1 + delta_lon2) / 2, 2));

            std::cout << "     image diagonal: \"" << diag << "m" << std::endl;

            niirs = metric.calculate_absolute(frame, diag, diag);
        }

        std::cout << "     NIIRS for \"" << path << "\": " << niirs << std::endl;
        csv << "\"" << path << "\"," << niirs << std::endl;

        putText(frame, "NIIRS: " + std::to_string(niirs), Point(40, 40), FONT_HERSHEY_DUPLEX, 1, (0xff, 0xff, 0, 0));

        std::cout << "Dedotated: " << opath << std::endl;
        imwrite(opath, frame);

    }
    csv.close();
    std::cout << "Results written to \"" << csv_path << "\"." << std::endl;
    return 0;
}

const int main_02(const int argc, const char** argv) {
    CNiirsMetric metric;

    cv::Mat frame = cv::imread(argv[1]);

    if(frame.empty()) {
        std::cout << "libniirs <image file>\nEstimates image quality / NIIRS degradation";
        return 0;
    }

    double niirs = metric.calculate(frame);
    std::cout << "NIIRS Degradation relative to ideal: -" << niirs << " (";
    int nirs = (int) niirs;
    switch(nirs) {
    case 0:
        std::cout << "PERFECT\n";
        break;
    case 1:
        std::cout << "GOOD\n";
        break;
    case 2:
        std::cout << "BAD\n";
        break;
    case 3:
        std::cout << "TERRIBLE\n";
        break;
    }
    return 0;
}

const int main_03(const int, const char**)
{
    CNiirsMetric metric;
    VideoCapture cap;
    Mat frame;

    cap.open(0);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR! Unable to open camera\n";

        return -1;
    }

    while (true)
    {
        cap >> frame;

        if (frame.empty())
            continue;

        imshow("FRAME", frame);

        const double niirs = metric.calculate_absolute(frame, 2, 24);

        std::cout << niirs << std::endl;

        waitKey();
    }
}




const int main(const int argc, const char** argv)
{
    return ENTRY_POINT(argc, argv);
}
