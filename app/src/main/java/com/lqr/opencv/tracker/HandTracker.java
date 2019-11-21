package com.lqr.opencv.tracker;

import android.text.TextUtils;

import org.opencv.core.Mat;

public class HandTracker {

    static {
        System.loadLibrary("HandTracker");
    }

    private static final int NULL = 0;
    private static long handModelNativeObj = NULL;
    private static long cascadeNativeObj = NULL;

    public static void initial(String handModelPath, String cascadePath) {
        handModelNativeObj = TextUtils.isEmpty(handModelPath) ? NULL : initHandModel(handModelPath);
        cascadeNativeObj = TextUtils.isEmpty(cascadePath) ? NULL : initCascadeClassifier(cascadePath);
    }

    public static void checkHand(Mat frameMat) {
        if (handModelNativeObj != NULL && cascadeNativeObj != NULL) {
            checkHand(frameMat.nativeObj, handModelNativeObj, cascadeNativeObj);
        } else if (handModelNativeObj != NULL) {
            checkHand(frameMat.nativeObj, handModelNativeObj);
        }
    }

    public static void destroy() {
        if (handModelNativeObj != NULL) {
            destroyHandModel(handModelNativeObj);
            handModelNativeObj = NULL;
        }
        if (cascadeNativeObj != NULL) {
            destroyCascadeClassifier(cascadeNativeObj);
            cascadeNativeObj = NULL;
        }
    }

    private native static long initHandModel(String handModelPath);

    private native static long initCascadeClassifier(String cascadePath);

    private native static void checkHand(long frameMat, long handModel);

    private native static void checkHand(long frameMat, long handModel, long cascade);

    private native static long destroyHandModel(long handModel);

    private native static long destroyCascadeClassifier(long cascade);
}
