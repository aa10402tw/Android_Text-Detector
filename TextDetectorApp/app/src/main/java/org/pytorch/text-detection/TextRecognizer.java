package org.pytorch.helloworld;

import org.pytorch.helloworld.MyUtils.*;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.MatOfPoint2f;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.*;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.icu.util.Output;
import android.os.Bundle;
import android.os.Debug;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.appcompat.app.AppCompatActivity;
import java.util.HashMap;
import java.util.Map;

public class TextRecognizer {
    public static String LogTag = "[Text Recognition]";
    public static int target_size = 400;
    public Module module;
    public boolean isDebug;
    public boolean isBilateralFilter = true;
    public int filterRadius = 5;

    // Decode
    public static List<Integer> INDEX_LIST;
    public static List<String> CHAR_LIST;
    public static Map<Integer, String> idx2char;
    public static Map<String, Integer> char2idx;
    public void InitDecorder() {
        INDEX_LIST = new ArrayList<Integer>();
        CHAR_LIST  = new ArrayList<String>();
        for(int i=0; i<38; i++) INDEX_LIST.add(i);
        for(int i=0; i<38; i++) {
            // if(i==16) CHAR_LIST.add("-");
            if(i==0) CHAR_LIST.add("[GO]");
            else if(i==1) CHAR_LIST.add("[END]");
            else if(2 <= i && i <= 11 ) CHAR_LIST.add(String.valueOf(i-2));
            else if(i>=12) CHAR_LIST.add(String.valueOf( (char) ('A' + (i-12))));
        }
        idx2char = new HashMap<>();
        for(int i=0; i<38; i++) idx2char.put(INDEX_LIST.get(i), CHAR_LIST.get(i));
        char2idx = new HashMap<>();
        for(int i=0; i<38; i++) char2idx.put(CHAR_LIST.get(i), INDEX_LIST.get(i));
    }

    /* --- Initialization functions --- */
    public TextRecognizer(Module module) {
        this.module = module;
        this.isDebug = false;
        InitDecorder();
    }
    public TextRecognizer(Module module, boolean isDebug) {
        this.module = module;
        this.isDebug = isDebug;
        InitDecorder();
    }
    public TextRecognizer(Module module, boolean isDebug, boolean isBilinearFilter) {
        this.module = module;
        this.isDebug = isDebug;
        isBilinearFilter = isBilinearFilter;
        InitDecorder();
    }

    /* --- Pre-Processing Functions --- */
    public Tensor GetInputImageTensor(Mat img) {
        // 1. Convert To Gray-scaled Image
        Mat img_gray = MyUtils.ToGrayScaledMat(img);
        if(isDebug) MyUtils.PrintMatStat(img_gray, "[Input Image]", LogTag );

        // 2. Resize Image
        img_gray.convertTo(img_gray, CvType.CV_32F);
        Imgproc.resize(img_gray, img_gray, new Size(100, 32), Imgproc.INTER_CUBIC);
        if(isDebug) MyUtils.PrintMatStat(img_gray, "[Resized Image]", LogTag );

        // 3. Convert Mat to Tensor range (0, 1)
        float[] data = new float[img_gray.rows()*img_gray.cols()];
        img_gray.get(0, 0, data);
        long[] input_img_shape = new long[]{1, 1, img_gray.rows(), img_gray.cols()};
        Tensor imageTensor = Tensor.fromBlob(data, input_img_shape);
        imageTensor = MyUtils.FloatTensorDivide(imageTensor, 255.0f);

        // 4. Normalize Tensor to range (-1, 1)
        if(isDebug) MyUtils.PrintTensorStat(imageTensor, "[Unnorm Image Tensor]", LogTag );
        imageTensor = MyUtils.FloatTensorSub(imageTensor, 0.5f);
        imageTensor = MyUtils.FloatTensorMultiply(imageTensor, 2.0f);
        if(isDebug) MyUtils.PrintTensorStat(imageTensor, "[Input Image Tensor]", LogTag );
        return imageTensor;
    }
    public Tensor GetInputTextTensor() {
        /* 3. Prepare Input Text Tensor */
        long[] text_for_pred = new long[26];
        long[] text_shape = new long[]{1, 26};
        Tensor textTensor = Tensor.fromBlob(text_for_pred, text_shape);
        if(isDebug) MyUtils.PrintLongTensor(textTensor, "[Input Text Tensor]", LogTag);
        return textTensor;
    }

    /* --- Post-Processing Functions --- */
    public String DecodeOutputTensor(Tensor outputTensor) {
        float[] out_data = outputTensor.getDataAsFloatArray();
        Mat out_score_mat = new Mat(26, 38, CvType.CV_32F);
        out_score_mat.put(0, 0, out_data);
        int[] preds_index = new int[26];
        String pred_string = "";
        for(int i=0; i<preds_index.length; i++) {
            Rect roi = new Rect(0, i, 38, 1);
            Mat submat = out_score_mat.submat(roi);
            MinMaxLocResult minMaxLocResult = Core.minMaxLoc(submat);
            preds_index[i] = (int)minMaxLocResult.maxLoc.x;
            String s = idx2char.get(preds_index[i]);
            if (s == "[END]")
                break;
            else
                pred_string += s;
        }
        return pred_string;
    }

    /* --- Core functions --- */
    public String GetText(Mat img) {
        if (isDebug) MyUtils.PrintFormat(LogTag);

        if(this.isBilateralFilter) {
            if (isDebug) MyUtils.PrintMatStat(img,"[Before Bilateral Filter]" , LogTag);
            img = MyUtils.BilateralFilter(img, filterRadius);
            if (isDebug) MyUtils.PrintMatStat(img,"[After Bilateral Filter]" , LogTag);
        }
        // 1. Prepare Input
        Tensor imageTensor = GetInputImageTensor(img);
        Tensor textTensor = GetInputTextTensor();

        /* 2. Feed Into Model */
        IValue output = this.module.forward(IValue.from(imageTensor), IValue.from(textTensor));
        Tensor outputTensor = output.toTensor();

        /* 3. Decode Output */
        String pred_string = DecodeOutputTensor(outputTensor);
        if (isDebug) Log.w(LogTag , "Predict:" + pred_string);
        return pred_string;
    }

}
