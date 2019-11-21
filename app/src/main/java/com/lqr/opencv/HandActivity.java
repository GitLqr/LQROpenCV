package com.lqr.opencv;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import com.lqr.opencv.tracker.HandTracker;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class HandActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    // System.loadLibrary("detection_based_tracker");
                    initHandTracker();
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private File loadFile(int rawFileId, String fileName) {
        File rawFile = null;
        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(rawFileId);
            File handDir = getDir("hand", Context.MODE_PRIVATE);
            rawFile = new File(handDir, fileName);
            FileOutputStream os = new FileOutputStream(rawFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rawFile;
    }

    private void initHandTracker() {
        File bwzFile = loadFile(R.raw.bwz, "bwz.jpg");
        File cascadeFile = loadFile(R.raw.haarcascade_frontalface_alt, "haarcascade_frontalface_alt.xml");
        // File cascadeFile = loadFile(R.raw.lbpcascade_frontalface, "lbpcascade_frontalface.xml");
        HandTracker.initial(bwzFile.getAbsolutePath(), cascadeFile.getAbsolutePath());
        bwzFile.delete();
        cascadeFile.delete();
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
        HandTracker.destroy();
    }

    private Mat mRgba;
    private Mat mRgb;

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        HandTracker.destroy();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        if (mRgb != null) {
            mRgb.release();
            mRgb = null;
        }
        mRgb = new Mat();
        Imgproc.cvtColor(mRgba, mRgb, Imgproc.COLOR_RGBA2RGB);
        HandTracker.checkHand(mRgb);
        return mRgb;
    }

}
