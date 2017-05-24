//
// opencv_test.cpp : define the program entry point.
//

#include <stdlib.h>
#include <stdio.h>

#include <iostream>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char * argv[])
{
    cv::CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this help message }"
        "{ verbose v      |      | show build configuration log }"
    );

    if (parser.has("help")) {
        parser.printMessage();
    }
    else if (parser.has("verbose")) {
        std::cout << cv::getBuildInformation().c_str() << std::endl;
    }
    else {
        std::cout << CV_VERSION << std::endl;
    }
    return 0;
}
