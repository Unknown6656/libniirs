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

// #include "libniirs.cpp"
#include "cniirsmetric.h"
#include "cniirsmetric.cpp"


using namespace cv;
namespace fs = std::experimental::filesystem;


const int main(const int argc, const char** argv)
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
        const Mat frame = imread(path.generic_string());

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
            std::string wld_path = path.generic_string() + "/../" + path.filename().generic_string();

            wld_path = wld_path.replace(wld_path.length() - 1 - path.extension().generic_string().length(), path.extension().generic_string().length(), ".wld");

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

            // 1�lat == 111.111km
            // 1�lon == 111.111 * cos(1rad lat)
            const double delta_lat1 = std::abs(coord[0] - coord[2]) * 111111;
            const double delta_lat2 = std::abs(coord[6] - coord[4]) * 111111;
            const double delta_lon1 = std::abs(coord[1] - coord[3]) * 111111 * std::cos((coord[0] - coord[2]) * 0.01745329251);
            const double delta_lon2 = std::abs(coord[7] - coord[5]) * 111111 * std::cos((coord[6] - coord[4]) * 0.01745329251);
            const double diag = std::sqrt(std::pow((delta_lat1 + delta_lat2) / 2, 2) + std::pow((delta_lon1 + delta_lon2) / 2, 2));

            std::cout << "     image diagonal: \"" << diag << "m" << std::endl;

            niirs = metric.calculate_absolute(frame, diag, diag);

        }

        std::cout << "     NIIRS for \"" << path << "\": " << pniirs << std::endl;
        csv << "\"" << path << "\"," << niirs << std::endl;

        putText(frame, "NIIRS: " + std::to_string(niirs), Point(), FONT_HERSHEY_SIMPLEX, 1, (0xff, 0, 0));
        imwrite(opath, frame);
    }

    csv.close();

    std::cout << "Results written to \"" << csv_path << "\"." << std::endl;

    return 0;
}
