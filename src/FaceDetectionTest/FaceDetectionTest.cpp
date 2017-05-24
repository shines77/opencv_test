//
// FaceDetectionTest.cpp : program entry.
//

#include <stdlib.h>
#include <stdio.h>
#include <tchar.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/imgproc/imgproc.hpp>

//
// See: http://blog.csdn.net/hujingshuang/article/details/47337707/
//
void hog_gamma_adjust_test()
{
    cv::Mat face = cv::imread("..\\..\\..\\data\\FaceDetectionTest\\hog\\test.bmp", cv::IMREAD_ANYCOLOR);
    cv::Mat face_rgb, face_gray;
    cv::Mat face_gamma, face_gamma_out;

#if 0
    printf("CV_8UC1 = %d, CV_8UC2 = %d, CV_8UC3 = %d, CV_8UC4 = %d\n",
        CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4);
    printf("CV_8SC1 = %d, CV_8SC2 = %d, CV_8SC3 = %d, CV_8SC4 = %d\n",
        CV_8SC1, CV_8SC2, CV_8SC3, CV_8SC4);
    printf("CV_16UC1 = %d, CV_16UC2 = %d, CV_16UC3 = %d, CV_16UC4 = %d\n",
        CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4);
    printf("CV_16SC1 = %d, CV_16SC2 = %d, CV_16SC3 = %d, CV_16SC4 = %d\n",
        CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4);
    printf("CV_32SC1 = %d, CV_32SC2 = %d, CV_32SC3 = %d, CV_32SC4 = %d\n",
        CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4);
    printf("CV_32F = %d, CV_32FC1 = %d, CV_32FC2 = %d, CV_32FC3 = %d, CV_32FC4 = %d\n",
        CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4);
    printf("\n");
#endif

    printf("face.type = %d, face.channels = %d, face.width = %d, face.height = %d\n\n",
        face.type(), face.channels(), face.size().width, face.size().height);

    cv::imshow("Face - 原图", face);

    // 转换为灰度图, RGBA, BGRA: CV_8UC4, RGB, BGR CV_8UC3, Gray: CV_8UC1
    if (face.type() == CV_8UC3) {
        cvtColor(face, face_gray, CV_BGR2GRAY, 1);
    }
    else if (face.type() == CV_8UC4) {
        cvtColor(face, face_gray, CV_BGRA2GRAY, 1);
    }
    else if (face.type() == CV_8UC1) {
        face.copyTo(face_gray);
    }

    printf("face_gray.type = %d, face_gray.channels = %d, face_gray.width = %d, face_gray.height = %d\n\n",
        face_gray.type(), face_gray.channels(), face_gray.size().width, face_gray.size().height);

    cv::imshow("Face - 灰度图", face_gray);

    // 转换成浮点
    face_gray.convertTo(face_gamma, CV_32FC1);
    // 浮点归一化: [0.0, 1.0]
    face_gamma *= 1.0 / 255.0;

    // gamma 校正: 平方根法
    cv::sqrt(face_gamma, face_gamma);
    printf("[Setp 1] face_gamma.type = %d, face_gamma.channels = %d, face_gamma.width = %d, face_gamma.height = %d\n\n",
        face_gamma.type(), face_gamma.channels(), face_gamma.size().width, face_gamma.size().height);

    cv::imshow("Gamma校正[1]", face_gamma);

    // 像素归一化: [0, 255]
    cv::normalize(face_gamma, face_gamma_out, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
    printf("[Setp 2] face_gamma_out.type = %d, face_gamma_out.channels = %d, face_gamma_out.width = %d, face_gamma_out.height = %d\n\n",
        face_gamma_out.type(), face_gamma_out.channels(), face_gamma_out.size().width, face_gamma_out.size().height);

    cv::imshow("Gamma校正[2]", face_gamma_out);

    cv::waitKey();
}

int main(int argc, char * argv[])
{
    hog_gamma_adjust_test();

    return 0;
}
