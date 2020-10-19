package org.pytorch.helloworld;

import java.util.ArrayList;
import java.util.Arrays;
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
import java.util.Arrays;

public class MyUtils {

    /* --- Log/Debug Functions --- */
    public static void PrintFormat(String tag) {
        Log.w(tag,String.format("%-25s %-22s |   Max    |   Min    |   Mean   |\n",
                "#---Object Label---#", "#---Type & Shape---#"));

    }
    public static void PrintMatStat(Mat mat) {
        Log.w("wen",  String.format("Matrix Shape:(%d,%d,%d) Max:%.4f, Min:%.4f, Mean:%.4f\n",
                mat.rows(), mat.cols(), mat.channels(), MyUtils.MatMax(mat), MyUtils.MatMin(mat), MyUtils.MatMean(mat)));
    }
    public static void PrintMatStat(Mat mat, String title) {
        if (mat.channels() == 1)
            Log.w("wen",  String.format("%s\t\t\t Matrix Shape:(%d,%d)\t Max:%.4f, Minx:%.4f, Mean:%.4f\n",
                    title, mat.rows(), mat.cols(), MatMax(mat), MatMin(mat), MatMean(mat)));
        else
            Log.w("wen",  String.format("%s\t\t\t Matrix Shape:(%d,%d,%d)\t Max:%.4f, Min:%.4f, Mean:%.4f\n",
                    title, mat.rows(), mat.cols(), mat.channels(), MatMax(mat), MatMin(mat), MatMean(mat)));
    }
    public static void PrintMatStat(Mat mat, String title, String tag) {
        Log.w(tag,  String.format("%-25s Matrix:%-15s |%-10.4f|%-10.4f|%-10.4f|\n",
                title, GetMatShapeStr(mat), MatMax(mat), MatMin(mat), MatMean(mat)));
    }
    public static void PrintTensorStat(Tensor tensor) {
        PrintTensorStat(tensor, "[No title]", "wen");
    }
    public static void PrintTensorStat(Tensor tensor, String title){
        PrintTensorStat(tensor, title, "wen");
    }
    public static void PrintTensorStat(Tensor tensor, String title, String tag){
        Log.w(tag, String.format("%-25s Tensor:%-15s |%-10.4f|%-10.4f|%-10.4f|\n",
                title, GetTensorShapeStr(tensor), TensorMax(tensor), TensorMin(tensor), TensorMean(tensor)));
    }
    public static void PrintLongTensor(Tensor tensor, String title, String tag) {
        long[] data = tensor.getDataAsLongArray();
        String s = "";
        if(data.length > 10)
            s += data[0] + "," +  data[1] + ","+ data[2] + ", ... "+ data[data.length-2] + ","+ data[data.length-1];
        else {
            for(int i=0; i<data.length; i++) {
                s += data[i];
                if(i<data.length)
                    s += ", ";
            }
        }

        Log.w(tag, String.format("%-25s Long Tensor:%-15s Data:[%s]\n",
                title, GetTensorShapeStr(tensor), s));
    }
    public static <T> void PrintList(List<T> list, String tag) {
        Log.w(tag, GetListString(list));
    }
    public static <T> String GetListString(List<T> list) {

        if (list == null) return "Not Process Yet";

        if(list.size() > 10) {
            return String.format("[%s, %s, %s, ... , %s, %s]",
                    list.get(0), list.get(1), list.get(2), list.get(list.size()-2), list.get(list.size()-1));
        }
        else{
            String s = "";
            for(int i=0; i<list.size(); i++) {
                s += list.get(i);
                if(i < list.size() - 1)
                    s += ", ";
            }
            return String.format("[%s]", s);
        }
    }

    /* --- OpenCV Mat Function  --- */
    public static String GetMatShapeStr(Mat mat) {
        if (mat.channels() == 1)
            return String.format("(%d,%d)", mat.rows(), mat.cols());
        else
            return String.format("(%d,%d,%d)", mat.rows(), mat.cols(), mat.channels());
    }
    public static double MatMax(Mat mat) {
        double max=0;
        for(int i=0; i<mat.rows(); i++) {
            for(int j=0; j<mat.cols(); j++) {
                double values[] = mat.get(i, j);
                for (int k = 0; k < values.length; k++)
                    max = Math.max(max, values[k]);
            }
        }
        return max;
    }
    public static double MatMin(Mat mat) {
        double min=255.0;
        for(int i=0; i<mat.rows(); i++) {
            for(int j=0; j<mat.cols(); j++) {
                double values[] = mat.get(i, j);
                for (int k = 0; k < values.length; k++)
                    min = Math.min(min, values[k]);
            }
        }
        return min;
    }
    public static double MatMean(Mat mat) {
        double sum=0.0;
        double count=0.0;
        for(int i=0; i<mat.rows(); i++) {
            for(int j=0; j<mat.cols(); j++) {
                double values[] = mat.get(i, j);
                for (int k = 0; k < mat.channels(); k++) {
                    sum += values[k];
                    count += 1.0;
                }
            }
        }
        return sum/count;
    }


    public static Mat BilateralFilter(Mat img, double radius) {
        Mat out = new Mat();
        Imgproc.bilateralFilter(ToColorMat(img), out, -1, radius, radius);
        return out;
    }
    public static Mat ResizeImg(Mat img, int target_size) {
        int img_h = img.rows();
        int img_w = img.cols();
        double ratio = (double)(target_size) / (double)Math.max(img_h, img_w);
        int target_w = (int)(img_w * ratio);
        int target_h = (int)(img_h * ratio);
        Mat img_resize = new Mat();
        Imgproc.resize(img, img_resize, new Size(target_w, target_h));
        return img_resize;
    }
    public static Mat CropImg(Mat img, Mat box) {
        Mat CropImg = new Mat();
        Rect roi_x = new Rect(0, 0, 1, 4);
        Mat box_x = box.submat(roi_x);
        Rect roi_y = new Rect(1, 0, 1, 4);
        Mat box_y = box.submat(roi_y);

        MinMaxLocResult minMaxLocResult_x =  Core.minMaxLoc(box_x);
        MinMaxLocResult minMaxLocResult_y =  Core.minMaxLoc(box_y);

        int x1 = (int) minMaxLocResult_x.minVal;
        int x2 = (int) minMaxLocResult_x.maxVal;
        int y1 = (int) minMaxLocResult_y.minVal;
        int y2 = (int) minMaxLocResult_y.maxVal;

        // Expand Box with boundary check
        int offset = 5;
        x1 = Math.max(0, x1-offset);
        x2 = Math.min(img.cols(), x2+offset);
        y1 = Math.max(0, y1-offset);
        y2 = Math.min(img.rows(), y2+offset);
        Rect crop_roi =  new Rect(x1, y1, x2-x1, y2-y1);
        return img.submat(crop_roi).clone();
    }

    public static Mat ToGrayScaledMat(Mat img) {
        Mat img_gray = new Mat();
        if(img.channels() == 4)
            Imgproc.cvtColor(img, img_gray, Imgproc.COLOR_BGRA2GRAY);
        else if(img.channels() == 3)
            Imgproc.cvtColor(img, img_gray, Imgproc.COLOR_BGR2GRAY);
        else if(img.channels() == 1)
            img_gray = img.clone();
        else
            Log.e("[ToGrayScaledMat]", "Invalid Input Image Channels" + img.channels());
        return img_gray;
    }
    public static Mat ToColorMat(Mat img) {
        Mat img_color = new Mat();
        if(img.channels() == 4)
            Imgproc.cvtColor(img, img_color, Imgproc.COLOR_BGRA2BGR);
        if(img.channels() == 3)
            img_color = img.clone();
        if(img.channels() == 1)
            Imgproc.cvtColor(img, img_color, Imgproc.COLOR_GRAY2BGR);
        return img_color;
    }

    /* --- PyTorch Tensor Functions --- */
    public static String GetTensorShapeStr(Tensor tensor) {
        String shapeText = "(";
        for(int i=0; i<tensor.shape().length; i++) {
            shapeText += tensor.shape()[i];
            if( i < tensor.shape().length-1)
                shapeText += ",";
            else
                shapeText += ")";
        }
        return shapeText;
    }
    public static double TensorMax(Tensor tensor) {
        float values[] = tensor.getDataAsFloatArray();
        double max=0.0;
        for(int i=0; i<values.length; i++)
            max = Math.max(max, (double)values[i]);
        return max;
    }
    public static double TensorMin(Tensor tensor) {
        float values[] = tensor.getDataAsFloatArray();
        double min=255.0;
        for(int i=0; i<values.length; i++)
            min = Math.min(min, (double)values[i]);
        return min;
    }
    public static double TensorMean(Tensor tensor) {
        float values[] = tensor.getDataAsFloatArray();
        double sum=0.0;
        for(int i=0; i<values.length; i++)
            sum += (double)values[i];
        return sum / values.length;
    }

    public static Tensor FloatTensorAdd(Tensor tensor, float x) {
        float[] data = tensor.getDataAsFloatArray();
        for(int i=0; i<data.length; i++)
            data[i] += x;
        return Tensor.fromBlob(data, tensor.shape());
    }
    public static Tensor FloatTensorSub(Tensor tensor, float x) {
        float[] data = tensor.getDataAsFloatArray();
        for(int i=0; i<data.length; i++)
            data[i] -= x;
        return Tensor.fromBlob(data, tensor.shape());
    }
    public static Tensor FloatTensorMultiply(Tensor tensor, float x) {
        float[] data = tensor.getDataAsFloatArray();
        for(int i=0; i<data.length; i++)
            data[i] *= x;
        return Tensor.fromBlob(data, tensor.shape());
    }
    public static Tensor FloatTensorDivide(Tensor tensor, float x) {
        float[] data = tensor.getDataAsFloatArray();
        for(int i=0; i<data.length; i++)
            data[i] /= x;
        return Tensor.fromBlob(data, tensor.shape());
    }

    /* Draw Functions */
    public static Mat DrawBoxes(Mat img, List<Mat> boxes) {
        // 1. Convert to BGR
        Mat img_display = ToColorMat(img);
        // 2. Draw Boxes
        for(int b=0; b<boxes.size(); b++)
            img_display = DrawBox(img_display, boxes.get(b));
        return img_display;
    }
    public static Mat DrawBox(Mat src, Mat box) {
        Mat dst = src.clone();
        int x1 = (int)box.get(0, 0)[0];
        int y1 = (int)box.get(0, 1)[0];
        int x2 = (int)box.get(2, 0)[0];
        int y2 = (int)box.get(2, 1)[0];
        Imgproc.rectangle(dst, new Point(x1, y1), new Point(x2, y2), new Scalar(255, 0, 0), 5);
        return dst;
    }
    public static Mat DrawBoxText(Mat img, Mat box, String text) {
        Mat img_display = img.clone();
        int x1 = (int)box.get(0, 0)[0];
        int y1 = (int)box.get(0, 1)[0];
        int x2 = (int)box.get(2, 0)[0];
        int y2 = (int)box.get(2, 1)[0];



        // Put Text
        int offset_y = 0;
        int offset_x = 0;
        int font_scale = (int) ((x2-x1) / text.length() / 10) ;

        // Draw Box
        Imgproc.rectangle(img_display, new Point(x1, y1), new Point(x2, y2), new Scalar(255, 0, 0), font_scale);
        Imgproc.putText (img_display, text, new Point(x1+offset_x, y1+offset_y),
                1, font_scale, new Scalar(255, 0, 0), font_scale);
        return img_display;

    }


}
