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
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/features2d.hpp>

#include "get_char.h"

void wait_any_key()
{
    printf("请先用鼠标点击任意一张图片, 再按下任意按键以关闭所以窗口 ...");
    cv::waitKey();
    printf("\n\n\n");
    cvDestroyAllWindows();
}

//
// See: http://blog.csdn.net/hujingshuang/article/details/47337707/
//
void HOG_gamma_adjust_test()
{
    cv::Mat face = cv::imread("..\\..\\..\\data\\FaceDetectionTest\\hog\\test.bmp", cv::IMREAD_ANYCOLOR);
    cv::Mat face_gray;
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

    cv::imshow("1.Face - 原图", face);

    // 转换为灰度图, RGBA, BGRA: CV_8UC4, RGB, BGR CV_8UC3, Gray: CV_8UC1
    if (face.type() == CV_8UC4) {
        cvtColor(face, face_gray, CV_BGRA2GRAY, 1);
    }
    else if (face.type() == CV_8UC3) {
        cvtColor(face, face_gray, CV_BGR2GRAY, 1);
    }
    else if (face.type() == CV_8UC2) {
        cvtColor(face, face_gray, CV_BGR5652GRAY, 1);
    }
    else if (face.type() == CV_8UC1) {
        face.copyTo(face_gray);
    }

    printf("face_gray.type = %d, face_gray.channels = %d, face_gray.width = %d, face_gray.height = %d\n\n",
        face_gray.type(), face_gray.channels(), face_gray.size().width, face_gray.size().height);

    cv::imshow("1.Face - 灰度图", face_gray);

    // 转换成浮点
    face_gray.convertTo(face_gamma, CV_32FC1);
    // 浮点归一化: [0.0, 1.0]
    face_gamma *= 1.0 / 255.0;

    // gamma 校正: 平方根法
    cv::sqrt(face_gamma, face_gamma);
    printf("[Setp 1] face_gamma.type = %d, face_gamma.channels = %d, face_gamma.width = %d, face_gamma.height = %d\n\n",
        face_gamma.type(), face_gamma.channels(), face_gamma.size().width, face_gamma.size().height);

    cv::imshow("1.Gamma校正[1]", face_gamma);

    // 像素归一化: [0, 255]
    cv::normalize(face_gamma, face_gamma_out, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
    printf("[Setp 2] face_gamma_out.type = %d, face_gamma_out.channels = %d, face_gamma_out.width = %d, face_gamma_out.height = %d\n\n",
        face_gamma_out.type(), face_gamma_out.channels(), face_gamma_out.size().width, face_gamma_out.size().height);

    cv::imshow("1.Gamma校正[2]", face_gamma_out);

    wait_any_key();
}

void HOG_gradient_test()
{
    cv::Mat face = cv::imread("..\\..\\..\\data\\FaceDetectionTest\\hog\\test.bmp", cv::IMREAD_ANYCOLOR);
    cv::Mat face_gray;
    cv::Mat face_gamma, face_gamma_out;

    cv::imshow("2.Face - 原图", face);

    // 转换为灰度图, RGBA, BGRA: CV_8UC4, RGB, BGR CV_8UC3, Gray: CV_8UC1
    if (face.type() == CV_8UC4) {
        cvtColor(face, face_gray, CV_BGRA2GRAY, 1);
    }
    else if (face.type() == CV_8UC3) {
        cvtColor(face, face_gray, CV_BGR2GRAY, 1);
    }
    else if (face.type() == CV_8UC2) {
        cvtColor(face, face_gray, CV_BGR5652GRAY, 1);
    }
    else if (face.type() == CV_8UC1) {
        face.copyTo(face_gray);
    }

    cv::imshow("2.Face - 灰度图", face_gray);

    // 转换成浮点
    face_gray.convertTo(face_gamma, CV_32FC1);
    // 浮点归一化: [0.0, 1.0]
    face_gamma *= 1.0 / 255.0;

    // gamma 校正: 平方根法
    cv::sqrt(face_gamma, face_gamma);
    // 像素归一化: [0, 255]
    cv::normalize(face_gamma, face_gamma_out, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("2.Gamma校正", face_gamma_out);

    cv::Mat gradient_out, theta_nor, theta_out;
    // 梯度
    cv::Mat gradient = cv::Mat::zeros(face_gamma_out.rows, face_gamma_out.cols, CV_32FC1);
    // 梯度角度
    cv::Mat theta = cv::Mat::zeros(face_gamma_out.rows, face_gamma_out.cols, CV_32FC1);

    for (int i = 1; i < face_gamma.rows - 1; i++) {
        for (int j = 1; j < face_gamma.cols - 1; j++) {
            float Gx, Gy;
            Gx = face_gamma.at<float>(i, j + 1) - face_gamma.at<float>(i, j - 1);
            Gy = face_gamma.at<float>(i + 1, j) - face_gamma.at<float>(i - 1, j);

            // 梯度模值
            gradient.at<float>(i, j) = sqrt(Gx * Gx + Gy * Gy);
            // 梯度角度: [-180°，180°]
            theta.at<float>(i, j) = float(atan2(Gy, Gx) * 180 / CV_PI);
        }
    }

    // 归一化: [0.0, 1.0]
    theta_nor = (theta + 180.0) / 360.0;

    // 归一化: [0, 255]
    cv::normalize(gradient, gradient_out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(theta_nor, theta_out, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("2.梯度图", gradient_out);
    cv::imshow("2.梯度角度", theta_out);
    
    wait_any_key();
}

void FAST_test()
{
    cv::Mat face = cv::imread("..\\..\\..\\data\\FaceDetectionTest\\fast\\test.png", cv::IMREAD_ANYCOLOR);
    cv::Mat face_gray;
    cv::Mat face_gamma, face_gamma_out;

    cv::imshow("3.Face - 原图", face);

    // 转换为灰度图, RGBA, BGRA: CV_8UC4, RGB, BGR CV_8UC3, Gray: CV_8UC1
    if (face.type() == CV_8UC4) {
        cvtColor(face, face_gray, CV_BGRA2GRAY, 1);
    }
    else if (face.type() == CV_8UC3) {
        cvtColor(face, face_gray, CV_BGR2GRAY, 1);
    }
    else if (face.type() == CV_8UC2) {
        cvtColor(face, face_gray, CV_BGR5652GRAY, 1);
    }
    else if (face.type() == CV_8UC1) {
        face.copyTo(face_gray);
    }

    cv::imshow("3.Face - 灰度图", face_gray);

    // 转换成浮点
    face_gray.convertTo(face_gamma, CV_32FC1);
    // 浮点归一化: [0.0, 1.0]
    face_gamma *= 1.0 / 255.0;

    // gamma 校正: 平方根法
    cv::sqrt(face_gamma, face_gamma);
    // 像素归一化: [0, 255]
    cv::normalize(face_gamma, face_gamma_out, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("3.Gamma校正", face_gamma_out);

    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(10, true, cv::FastFeatureDetector::TYPE_9_16);

    std::vector<cv::KeyPoint> key_points;
    detector->detect(face_gamma_out, key_points);

    cv::Mat fast_9_16, fast_7_12;
    cv::drawKeypoints(face_gamma_out, key_points, fast_9_16, cv::Scalar::all(-1), 0);

    cv::imshow("3.FAST关键点9", fast_9_16);

    detector = cv::FastFeatureDetector::create(10, true, cv::FastFeatureDetector::TYPE_7_12);
    detector->detect(face_gamma_out, key_points);
    cv::drawKeypoints(face_gamma_out, key_points, fast_7_12, cv::Scalar::all(-1), 0);

    cv::imshow("3.FAST关键点7", fast_7_12);

    wait_any_key();
}

//
// See: http://blog.csdn.net/morewindows/article/details/8239625
//

void canny_test()
{
    IplImage * src_gray = cvLoadImage("..\\..\\..\\data\\FaceDetectionTest\\canny\\yaoyao.png", CV_LOAD_IMAGE_GRAYSCALE);
    IplImage * canny10 = cvCreateImage(cvGetSize(src_gray), IPL_DEPTH_8U, 1);
    IplImage * canny20 = cvCreateImage(cvGetSize(src_gray), IPL_DEPTH_8U, 1);
    IplImage * canny30 = cvCreateImage(cvGetSize(src_gray), IPL_DEPTH_8U, 1);
    IplImage * canny40 = cvCreateImage(cvGetSize(src_gray), IPL_DEPTH_8U, 1);
    IplImage * canny50 = cvCreateImage(cvGetSize(src_gray), IPL_DEPTH_8U, 1);
    IplImage * canny60 = cvCreateImage(cvGetSize(src_gray), IPL_DEPTH_8U, 1);
    IplImage * canny80 = cvCreateImage(cvGetSize(src_gray), IPL_DEPTH_8U, 1);
    IplImage * canny100 = cvCreateImage(cvGetSize(src_gray), IPL_DEPTH_8U, 1);

    cvShowImage("原图", src_gray); 

    // canny 边缘检测
    double threshold = 10.0;
    cvCanny(src_gray, canny10, threshold, threshold * 3, 3);
    cvShowImage("threshold=10", canny10);

    threshold = 20.0;
    cvCanny(src_gray, canny20, threshold, threshold * 3, 3);
    cvShowImage("threshold=20", canny20);

    threshold = 30.0;
    cvCanny(src_gray, canny30, threshold, threshold * 3, 3);
    cvShowImage("threshold=30", canny30);

    threshold = 40.0;
    cvCanny(src_gray, canny40, threshold, threshold * 3, 3);
    cvShowImage("threshold=40", canny40);

    threshold = 50.0;
    cvCanny(src_gray, canny50, threshold, threshold * 3, 3);
    cvShowImage("threshold=50", canny50);

    threshold = 60.0;
    cvCanny(src_gray, canny60, threshold, threshold * 3, 3);
    cvShowImage("threshold=60", canny60);

    threshold = 80.0;
    cvCanny(src_gray, canny80, threshold, threshold * 3, 3);
    cvShowImage("threshold=80", canny80);

    threshold = 100.0;
    cvCanny(src_gray, canny100, threshold, threshold * 3, 3);
    cvShowImage("threshold=100", canny100);

    wait_any_key();

    if (src_gray)
        cvReleaseImage(&src_gray);
    if (canny10)
        cvReleaseImage(&canny10);
    if (canny20)
        cvReleaseImage(&canny20);
    if (canny30)
        cvReleaseImage(&canny30);
    if (canny40)
        cvReleaseImage(&canny40);
    if (canny50)
        cvReleaseImage(&canny50);
    if (canny60)
        cvReleaseImage(&canny60);
    if (canny80)
        cvReleaseImage(&canny80);
    if (canny100)
        cvReleaseImage(&canny100);
}

int get_user_choice(int lang_id, const char * display_text, const char * tips_format_text_,
                    int min_value, int max_value, int default_value)
{
    const int exit_value = GETCH_EXIT_PROGRAM;
    int input_value      = GETCH_DEFUALT_VALUE;
    int tmp_value;

    const char * tips_format_text = "你的选择是: ";

    printf("%s", display_text);
    if (tips_format_text_ == NULL)
        printf(tips_format_text, exit_value);
    else
        printf(tips_format_text_, exit_value);
    if (default_value == -1)
        printf("?");
    else
        printf("%d", default_value);

    int nchar;
    do {
        nchar = jm_getch();
        if (nchar == MM_VT_KEY_RETURN) {
            if (input_value == GETCH_DEFUALT_VALUE)
                input_value = default_value;
            break;
        }
        else if (nchar >= '0' && nchar <= '9') {
            tmp_value = nchar - '0';
            if (tmp_value >= min_value && tmp_value <= max_value) {
                input_value = tmp_value;
                printf("\x08%d", input_value);
                fflush(stdout);
            }
            else {
                // 如果输入的字符不在指定的范围, 显示一下(500毫秒)立刻恢复最后一次正确的选项值
                printf("\x08%d", tmp_value);
                fflush(stdout);

                // 休眠 500 毫秒
                jm_sleep(500);

                if (input_value != GETCH_DEFUALT_VALUE)
                    printf("\x08%d", input_value);
                else
                    printf("\x08%d", default_value);
                fflush(stdout);
            }
        }
    } while (1);

    printf("\n\n");
    return input_value;
}

int get_test_func(int default_id = -1)
{
    const char * display_text =
        "请选择测试编号:\n"
        "\n"
        "[1] = HOG gamma 校正.\n"
        "[2] = HOG 梯度计算.\n"
        "[3] = FAST 特征关键点.\n"
        "[4] = Canny 边缘查找.\n"
        "\n"
        "[0] = 退出程序.\n\n"
        ""
        "输入你的选择后, 并按 [回车键] 以确定选择.\n\n";

    const char * tips_format_text = "你的选择是: ";

    return get_user_choice(0, display_text, tips_format_text, 0, 4, default_id);
}

int main(int argc, char * argv[])
{
Retry:
    int test_func_id = get_test_func();

    switch (test_func_id) {
    case 0:
        cvDestroyAllWindows();
        goto Exit;

    case 1:
        // 演示 gamma 校正
        HOG_gamma_adjust_test();
        break;

    case 2:
        // 演示计算 HOG 梯度
        HOG_gradient_test();
        break;

    case 3:
        // 演示 FAST 特征关键点 (SFIT opencv 2.x 才有, 3.x 版本没了)
        FAST_test();
        break;

    case 4:
        // Canny 边缘检测
        canny_test();
        break;

    default:
        printf("\n");
        printf("无效的输入, 请重新选择.\n");
        printf("\n\n");
        break;
    } 

    goto Retry;

Exit:
    return 0;
}
