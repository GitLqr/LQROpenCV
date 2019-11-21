#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <stack>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/objdetect.hpp>

#include <android/log.h>

#define LOG_TAG "FaceDetection/HandTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;

class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector {
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector)
            : IDetector(), Detector(detector) {
        LOGD("CascadeDetectorAdapter::Detect::Detect");
        CV_Assert(detector);
    }

    void detect(const cv::Mat &image, std::vector<cv::Rect> &objects) {
        LOGD("CascadeDetectorAdapter::Detect: begin");
        LOGD("CascadeDetectorAdapter::Detect: scaleFactor=%.2f, minNeighbours=%d, minObjSize=(%dx%d), maxObjSize=(%dx%d)",
             scaleFactor, minNeighbours, minObjSize.width, minObjSize.height, maxObjSize.width,
             maxObjSize.height);
        Detector->detectMultiScale(image, objects, scaleFactor, minNeighbours, 0, minObjSize,
                                   maxObjSize);
        LOGD("CascadeDetectorAdapter::Detect: end");
    }

    virtual ~CascadeDetectorAdapter() {
        LOGD("CascadeDetectorAdapter::Detect::~Detect");
    }

private:
    CascadeDetectorAdapter();

    cv::Ptr<cv::CascadeClassifier> Detector;
};

struct DetectorAgregator {
    cv::Ptr<CascadeDetectorAdapter> mainDetector;
    cv::Ptr<CascadeDetectorAdapter> trackingDetector;

    cv::Ptr<DetectionBasedTracker> tracker;

    DetectorAgregator(cv::Ptr<CascadeDetectorAdapter> &_mainDetector,
                      cv::Ptr<CascadeDetectorAdapter> &_trackingDetector) :
            mainDetector(_mainDetector),
            trackingDetector(_trackingDetector) {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);

        DetectionBasedTracker::Parameters DetectorParams;
        tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};

//8邻接种子算法，并返回每块区域的边缘框
void Seed_Filling(const cv::Mat &binImg, cv::Mat &labelImg, int &labelNum, int(&ymin)[20],
                  int(&ymax)[20], int(&xmin)[20], int(&xmax)[20])   //种子填充法
{
    if (binImg.empty() ||
        binImg.type() != CV_8UC1) {
        return;
    }

    labelImg.release();
    binImg.convertTo(labelImg, CV_32SC1);
    int label = 1;
    int rows = binImg.rows - 1;
    int cols = binImg.cols - 1;
    for (int i = 1; i < rows - 1; i++) {
        int *data = labelImg.ptr<int>(i);
        for (int j = 1; j < cols - 1; j++) {
            if (data[j] == 1) {
                std::stack<std::pair<int, int>> neighborPixels;
                neighborPixels.push(std::pair<int, int>(j, i));     // 像素位置: <j,i>
                ++label;  // 没有重复的团，开始新的标签
                ymin[label] = i;
                ymax[label] = i;
                xmin[label] = j;
                xmax[label] = j;
                while (!neighborPixels.empty()) {
                    std::pair<int, int> curPixel = neighborPixels.top(); //如果与上一行中一个团有重合区域，则将上一行的那个团的标号赋给它
                    int curX = curPixel.first;
                    int curY = curPixel.second;
                    labelImg.at<int>(curY, curX) = label;

                    neighborPixels.pop();

                    if ((curX > 0) && (curY > 0) && (curX < (cols - 1)) && (curY < (rows - 1))) {
                        if (labelImg.at<int>(curY - 1, curX) == 1)                     //上
                        {
                            neighborPixels.push(std::pair<int, int>(curX, curY - 1));
                            //ymin[label] = curY - 1;
                        }
                        if (labelImg.at<int>(curY + 1, curX) == 1)                        //下
                        {
                            neighborPixels.push(std::pair<int, int>(curX, curY + 1));
                            if ((curY + 1) > ymax[label])
                                ymax[label] = curY + 1;
                        }
                        if (labelImg.at<int>(curY, curX - 1) == 1)                     //左
                        {
                            neighborPixels.push(std::pair<int, int>(curX - 1, curY));
                            if ((curX - 1) < xmin[label]) xmin[label] = curX - 1;
                        }
                        if (labelImg.at<int>(curY, curX + 1) == 1)                     //右
                        {
                            neighborPixels.push(std::pair<int, int>(curX + 1, curY));
                            if ((curX + 1) > xmax[label])
                                xmax[label] = curX + 1;
                        }
                        if (labelImg.at<int>(curY - 1, curX - 1) == 1)                   //左上
                        {
                            neighborPixels.push(std::pair<int, int>(curX - 1, curY - 1));
                            //ymin[label] = curY - 1;
                            if ((curX - 1) < xmin[label]) xmin[label] = curX - 1;
                        }
                        if (labelImg.at<int>(curY + 1, curX + 1) == 1)                   //右下
                        {
                            neighborPixels.push(std::pair<int, int>(curX + 1, curY + 1));
                            if ((curY + 1) > ymax[label])
                                ymax[label] = curY + 1;
                            if ((curX + 1) > xmax[label])
                                xmax[label] = curX + 1;

                        }
                        if (labelImg.at<int>(curY + 1, curX - 1) == 1)                    //左下
                        {
                            neighborPixels.push(std::pair<int, int>(curX - 1, curY + 1));
                            if ((curY + 1) > ymax[label])
                                ymax[label] = curY + 1;
                            if ((curX - 1) < xmin[label]) xmin[label] = curX - 1;
                        }
                        if (labelImg.at<int>(curY - 1, curX + 1) == 1)                    //右上
                        {
                            neighborPixels.push(std::pair<int, int>(curX + 1, curY - 1));
                            //ymin[label] = curY - 1;
                            if ((curX + 1) > xmax[label])
                                xmax[label] = curX + 1;

                        }
                    }
                }
            }
        }
    }
    labelNum = label - 1;

}

class WatershedSegmenter {
private:
    cv::Mat markers;
public:
    void setMarkers(const cv::Mat &markerImage) {

        // Convert to image of ints
        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(const cv::Mat &image) {

        // Apply watershed
        cv::watershed(image, markers);
        return markers;
    }

    // Return result in the form of an image
    cv::Mat getSegmentation() {

        cv::Mat tmp;
        // all segment with label higher than 255
        // will be assigned value 255
        markers.convertTo(tmp, CV_8U);
        return tmp;
    }

    // Return watershed in the form of an image
    cv::Mat getWatersheds() {
        cv::Mat tmp;
        markers.convertTo(tmp, CV_8U, 255, 255);
        return tmp;
    }
};

void getSkinSection(Mat &frame, Mat &handModel, vector<Rect> &skinSection) {
    Mat binImage, tmp;
    Mat Y, Cr, Cb;
    vector<Mat> channels;

    // cvtColor(&frame, &frame, CV_RGBA2RGB);

    //转换颜色空间，并分割通道
    // cvtColor(frame, binImage, CV_BGR2GRAY);
    cvtColor(frame, binImage, CV_RGB2GRAY);
    frame.copyTo(tmp);

    // cvtColor(tmp, tmp, CV_BGR2YCrCb);
    cvtColor(tmp, tmp, CV_RGB2YCrCb);
    split(tmp, channels);
    Cr = channels.at(1);
    Cb = channels.at(2);

    //肤色检测，输出二值图像
    for (int j = 1; j < Cr.rows - 1; j++) {
        uchar *currentCr = Cr.ptr<uchar>(j);
        uchar *currentCb = Cb.ptr<uchar>(j);
        uchar *current = binImage.ptr<uchar>(j);
        for (int i = 1; i < Cb.cols - 1; i++) {
            if ((currentCr[i] > 140) && (currentCr[i] < 170) && (currentCb[i] > 77) &&
                (currentCb[i] < 123))
                current[i] = 255;
            else
                current[i] = 0;
        }
    }

    //形态学处理
    //dilate(binImage, binImage, Mat());
    dilate(binImage, binImage, Mat());

    //分水岭算法
    cv::Mat fg;
    cv::erode(binImage, fg, cv::Mat(), cv::Point(-1, -1), 6);
    // Identify image pixels without objects
    cv::Mat bg;
    cv::dilate(binImage, bg, cv::Mat(), cv::Point(-1, -1), 6);
    cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);
    // Show markers image
    cv::Mat markers(binImage.size(), CV_8U, cv::Scalar(0));
    markers = fg + bg;
    // Create watershed segmentation object
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    segmenter.process(frame);
    Mat waterShed;
    waterShed = segmenter.getWatersheds();

    //imshow("watershed", waterShed);
    //获得区域边框
    threshold(waterShed, waterShed, 1, 1, THRESH_BINARY_INV);

    //8向种子算法，给边框做标记
    Mat labelImg;
    int label, ymin[20], ymax[20], xmin[20], xmax[20];
    Seed_Filling(waterShed, labelImg, label, ymin, ymax, xmin, xmax);

    //根据标记，对每块候选区就行缩放，并与模板比较
    Size dsize = Size(handModel.cols, handModel.rows);

    if (dsize.width <= 0)
        return;

    float simi[20];
    for (int i = 0; i < label; i++) {
        simi[i] = 1;
        if (((xmax[2 + i] - xmin[2 + i]) > 50) && ((ymax[2 + i] - ymin[2 + i]) > 50)) {
            //rectangle(frame, Point(xmin[2 + i], ymin[2 + i]), Point(xmax[2 + i], ymax[2 + i]), Scalar::all(255), 2, 8, 0);
            Mat rROI = Mat(dsize, CV_8UC1);
            resize(Cr(Rect(xmin[2 + i], ymin[2 + i], xmax[2 + i] - xmin[2 + i],
                           ymax[2 + i] - ymin[2 + i])), rROI, dsize);
            Mat result;
            matchTemplate(rROI, handModel, result, CV_TM_SQDIFF_NORMED);
            simi[i] = result.ptr<float>(0)[0];
            //cout << simi[i] << endl;
        }
    }

    //统计一下区域中的肤色区域比例
    float fuseratio[20];
    for (int k = 0; k < label; k++) {
        fuseratio[k] = 1;
        if (((xmax[2 + k] - xmin[2 + k]) > 50) && ((ymax[2 + k] - ymin[2 + k]) > 50)) {
            int fusepoint = 0;
            for (int j = ymin[2 + k]; j < ymax[2 + k]; j++) {
                uchar *current = binImage.ptr<uchar>(j);
                for (int i = xmin[2 + k]; i < xmax[2 + k]; i++) {
                    if (current[i] == 255)
                        fusepoint += 1;
                }
            }
            fuseratio[k] = float(fusepoint) /
                           ((xmax[2 + k] - xmin[2 + k]) * (ymax[2 + k] - ymin[2 + k]));
            //cout << fuseratio[k] << endl;
        }
    }

    //给符合阈值条件的位置画框
    for (int i = 0; i < label; i++) {
        if ((simi[i] < 0.02) && (fuseratio[i] < 0.65)) {
            // rectangle(frame, Point(xmin[2 + i], ymin[2 + i]), Point(xmax[2 + i], ymax[2 + i]), Scalar::all(255), 2, 8, 0);
            skinSection.push_back(Rect(xmin[2 + i], ymin[2 + i], xmax[2 + i] - xmin[2 + i],
                                       ymax[2 + i] - ymin[2 + i]));
        }
    }

    // imshow("frame", frame); // 原图 + 框
    //processor.writeNextFrame(frame);
    // imshow("test", binImage); // 黑白图
}

void
getFaceSection(Mat &image, DetectorAgregator &detectorAgregator, vector<cv::Rect> &faceSection) {
    cv::Mat imageGray;
    cvtColor(image, imageGray, COLOR_BGR2GRAY);

    detectorAgregator.tracker->process(imageGray);
    detectorAgregator.tracker->getObjects(faceSection);
    Mat faces = Mat(faceSection, true);
    LOGD("faceSection.size() = %d", faceSection.size());
    LOGD("faces.size() = %d", faces.size);
}

//void getFaceSection(Mat &image, CascadeClassifier &cascade, vector<cv::Rect> &faceSection) {
//    double scale = 1;
//    std::vector<cv::Rect> faces;
//    cv::Mat gray;
//
//    // --Detection
//
//    // Read Opencv Detection Bbx
//    //cv::Mat smallImg(cvRound(image.rows / scale), cvRound(image.cols / scale), CV_8UC1); //cvRound对double型数据进行四舍五入
//    //cv::resize(image, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
//    //cv::equalizeHist(smallImg, smallImg);                                              //equalizeHist提高图像的高度和
//
//    cvtColor(image, gray, COLOR_BGR2GRAY);
//
//    // --Detection
//    cascade.detectMultiScale(gray, faces, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING
//            /*|CV_HAAR_FIND_BIGGEST_OBJECT
//            |CV_HAAR_DO_ROUGH_SEARCH*/
//            //| CV_HAAR_SCALE_IMAGE
//            , cv::Size(100, 100));
//
//    for (std::vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end(); r++) {
//        //cv::Rect rect(0, 0, 0, 0);
//
//        //rect.x = int(r->x*scale);
//        //rect.y = int(r->y*scale);
//        //rect.width = int((r->width - 1)*scale);
//        //rect.height = int((r->height - 1)*scale);
//
//        //cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 3, 8);
//
//        faceSection.push_back(
//                cv::Rect(int(r->x * scale), int(r->y * scale), int((r->width - 1) * scale),
//                         int((r->height - 1) * scale)));
//    }
//
//    faces.clear();
//}

bool isOverlap(cv::Rect *rect1, cv::Rect *rect2) {
    int startX1 = rect1->x;
    int startY1 = rect1->y;
    int endX1 = startX1 + rect1->width;
    int endY1 = startY1 + rect1->height;

    int startX2 = rect2->x;
    int startY2 = rect2->y;
    int endX2 = startX2 + rect2->width;
    int endY2 = startY2 + rect2->height;

    return !(endY2 < startY1 || endY1 < startY2 || startX1 > endX2 || startX2 > endX1);
};

bool isOverlap(int x1, int y1, int width1, int height1, int x2, int y2, int width2, int height2) {
    int startX1 = x1;
    int startY1 = y1;
    int endX1 = startX1 + width1;
    int endY1 = startY1 + height1;

    int startX2 = x2;
    int startY2 = y2;
    int endX2 = startX2 + width2;
    int endY2 = startY2 + height2;

    return !(endY2 < startY1 || endY1 < startY2 || startX1 > endX2 || startX2 > endX1);
}

void getHandSection(vector<cv::Rect> &pifuSection, vector<cv::Rect> &faceSection,
                    vector<cv::Rect> &handSectionOut) {
    bool bOver = false;
    for (std::vector<cv::Rect>::iterator rp = pifuSection.begin(); rp != pifuSection.end(); rp++) {
        bOver = false;
        for (std::vector<cv::Rect>::iterator rf = faceSection.begin();
             rf != faceSection.end(); rf++) {
            cv::Rect *p1 = &(*rp);
            cv::Rect *p2 = &(*rf);
            if (isOverlap(p1, p2) == true) {
                bOver = true;
                break;
            }
        }

        if (bOver == false) {
            handSectionOut.push_back(*rp);
        }
    }
}

void drawSection(Mat &frame, vector<cv::Rect> &section) {
    int count = (int) section.size();
    if (count > 0) {
        for (int i = 0; i < count; i++) {
            cv::Rect rect = section[i];
            rectangle(frame, Point(rect.x, rect.y),
                      Point(rect.x + rect.width, rect.y + rect.height),
                      Scalar::all(255), 2, 8, 0);
        }
    }
}

void checkHand(Mat &frame, Mat &handModel) {
    vector<cv::Rect> skinSection;
    getSkinSection(frame, handModel, skinSection);
    //给符合阈值条件的位置画框
    drawSection(frame, skinSection);
}

void checkHand(Mat &frame, Mat &handModel, DetectorAgregator &detectorAgregator) {
//    vector<cv::Rect> skinSection;
//    vector<cv::Rect> faceSection;
//    vector<cv::Rect> handSection;
//
//    getFaceSection(frame, detectorAgregator, faceSection);
//    getSkinSection(frame, handModel, skinSection);
//    getHandSection(skinSection, faceSection, handSection);
//
//    drawSection(frame, handSection);
//
//    skinSection.clear();
//    faceSection.clear();
//    handSection.clear();

    vector<cv::Rect> faceSection;
    getFaceSection(frame, detectorAgregator, faceSection);
    faceSection.clear();
}

//void checkHand(Mat &frame, Mat &handModel, CascadeClassifier &cascade) {
//    vector<cv::Rect> skinSection;
//    vector<cv::Rect> faceSection;
//    vector<cv::Rect> handSection;
//
//    getFaceSection(frame, cascade, faceSection);
//    getSkinSection(frame, handModel, skinSection);
//    getHandSection(skinSection, faceSection, handSection);
//
//    drawSection(frame, handSection);
//
//    skinSection.clear();
//    faceSection.clear();
//    handSection.clear();
//}


#include "HandTracker.h"

extern "C"
JNIEXPORT jlong JNICALL Java_com_lqr_opencv_tracker_HandTracker_initHandModel
        (JNIEnv *env, jclass jcls, jstring jhandModelPath) {

    const char *handModelPath = env->GetStringUTFChars(jhandModelPath, NULL);
    Mat handModel = imread(handModelPath, CV_8UC1);
    env->ReleaseStringUTFChars(jhandModelPath, handModelPath);

    Mat *nativeMat = new Mat();
    handModel.copyTo(*nativeMat);
    return (long) nativeMat;
}

extern "C"
JNIEXPORT jlong JNICALL Java_com_lqr_opencv_tracker_HandTracker_initCascadeClassifier
        (JNIEnv *env, jclass jcls, jstring jcascadePath) {

//    const char *cascadePath = env->GetStringUTFChars(jcascadePath, NULL);
//    CascadeClassifier *cascade = new CascadeClassifier(cascadePath);
//    env->ReleaseStringUTFChars(jcascadePath, cascadePath);
//    return (long) cascade;

    const char *jnamestr = env->GetStringUTFChars(jcascadePath, NULL);
    string stdFileName(jnamestr);
    jlong result = 0;

    int faceSize = 192;
    cv::Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(
            makePtr<CascadeClassifier>(stdFileName));
    cv::Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(
            makePtr<CascadeClassifier>(stdFileName));
    result = (jlong) new DetectorAgregator(mainDetector, trackingDetector);
    if (faceSize > 0) {
        mainDetector->setMinObjectSize(Size(faceSize, faceSize));
        //trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
    }
    ((DetectorAgregator *) result)->tracker->run();

    return result;
}
extern "C"
JNIEXPORT void JNICALL Java_com_lqr_opencv_tracker_HandTracker_checkHand__JJ
        (JNIEnv *env, jclass jcls, jlong frameMat, jlong handModel) {

    try {
        Mat frameMatNative = (*(Mat *) frameMat);
        Mat handMatNative = (*(Mat *) handModel);

        checkHand(frameMatNative, handMatNative);
    }
    catch (cv::Exception &e) {
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je)
            je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
    }
}

extern "C"
JNIEXPORT void JNICALL Java_com_lqr_opencv_tracker_HandTracker_checkHand__JJJ
        (JNIEnv *env, jclass jcls, jlong frameMat, jlong handModel, jlong cascade) {

    try {
        Mat frameMatNative = (*(Mat *) frameMat);
        Mat handMatNative = (*(Mat *) handModel);

//        CascadeClassifier cascadeNative = (*((CascadeClassifier *) cascade));
//        checkHand(frameMatNative, handMatNative, cascadeNative);

        DetectorAgregator detectorAgregator = (*((DetectorAgregator *) cascade));
        checkHand(frameMatNative, handMatNative, detectorAgregator);
    }
    catch (cv::Exception &e) {
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je)
            je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
    }
}

extern "C"
JNIEXPORT jlong JNICALL Java_com_lqr_opencv_tracker_HandTracker_destroyHandModel
        (JNIEnv *env, jclass jcls, jlong handModel) {
    if (handModel != NULL) {
        ((Mat *) handModel)->release();
    }
}

extern "C"
JNIEXPORT jlong JNICALL Java_com_lqr_opencv_tracker_HandTracker_destroyCascadeClassifier
        (JNIEnv *env, jclass jcls, jlong cascade) {
    if (cascade != NULL) {
        ((DetectorAgregator *) cascade)->tracker->stop();
        delete (((CascadeClassifier *) cascade));
    }
}