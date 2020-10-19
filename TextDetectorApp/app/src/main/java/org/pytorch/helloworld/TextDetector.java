package org.pytorch.helloworld;

import java.util.ArrayList;
import java.util.List;
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
import java.util.Arrays;
public class TextDetector {

    public static String LogTag = "[Text Detection]";
    public static int target_size = 400;
    public Module module;
    public boolean isDebug;

    // Parameters
    static double text_threshold = 0.7;
    static double low_link = 0.4;
    static double low_text = 0.4;

    /* --- Initialization --- */
    public TextDetector(Module module) {
        this.module = module;
        this.isDebug = false;
    }
    public TextDetector(Module module, boolean isDebug) {
        this.module = module;
        this.isDebug = isDebug;
    }

    /* --- Convert PyTorch Tensor To OpenCV Mat --- */
    public Mat OutputTensorToScoreTextMat(Tensor outputTensor) {
        int rows = (int)outputTensor.shape()[2];
        int cols = (int)outputTensor.shape()[3];
        float[] data = outputTensor.getDataAsFloatArray();
        Mat score_text = new Mat(rows, cols, CvType.CV_32F);
        score_text.put(0, 0, Arrays.copyOfRange(data,0, rows*cols));
        return score_text;
    }
    public Mat OutputTensorToScoreLinkMat(Tensor outputTensor) {
        int rows = (int)outputTensor.shape()[2];
        int cols = (int)outputTensor.shape()[3];
        float[] data = outputTensor.getDataAsFloatArray();
        Mat score_link = new Mat(rows, cols, CvType.CV_32F);
        score_link.put(0, 0, Arrays.copyOfRange(data,rows*cols, data.length));
        return score_link;
    }

    /* --- Define Return Class --- */
    private static class DetBoxCoreValues {
        List<Mat> det;
        Mat labels;
        List<Integer> mapper;
        public DetBoxCoreValues(List<Mat> det, Mat labels, List<Integer> mapper) {
            this.det = det;
            this.labels = labels;
            this.mapper = mapper;
        }

    }

    /* --- Post-Processing Functions --- */
    public boolean TextScoreMapLessThanThreshold(Mat TextScoreMap, Mat labels, int k, double text_threshold){
        double max = 0.0;
        for(int i=0; i<TextScoreMap.rows(); i++)
            for(int j=0; j<TextScoreMap.cols(); j++)
                if ( (int)(labels.get(i, j)[0]) == k )
                    max = Math.max(max, TextScoreMap.get(i, j)[0]);

        if(max < text_threshold)
            return true;
        else
            return false;
    }
    public Mat CreateSegMap(Mat score_text,  Mat labels, int k) {
        Mat segmap = new Mat(score_text.rows(), score_text.cols(), CvType.CV_8U);
        for(int i=0; i<segmap.rows(); i++) {
            for(int j=0; j<segmap.cols(); j++) {
                if ((int) (labels.get(i, j)[0]) == k)
                    segmap.put(i, j, 255);
                else
                    segmap.put(i, j, 0);
            }
        }
        return segmap;
    }
    public Mat RemoveLinkArea(Mat segmap, Mat threshold_score_text, Mat threshold_score_link) {
        Mat newSegmap = segmap.clone();
        for(int i=0; i<segmap.rows(); i++) {
            for (int j = 0; j < segmap.cols(); j++) {
                int link_score = (int) (threshold_score_link.get(i, j)[0]);
                int text_score = (int) (threshold_score_text.get(i, j)[0]);
                if (link_score == 1 && text_score == 0)
                    newSegmap.put(i, j, 0);
            }
        }
        return newSegmap;
    }
    public Mat DilateSegMap(Mat segmap, int niter, int sx, int ex, int sy, int ey) {
        Mat newSegmap = segmap.clone();
        Size kernelSize = new Size(1 + niter, 1 + niter);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernelSize);

        Rect roi = new Rect(sx, sy, ex-sx, ey-sy);
        Mat submat = newSegmap.submat(roi);
        Imgproc.dilate(submat, submat, kernel);
        return newSegmap;
    }
    public Mat AlignBox(Mat box, Mat nonzero_idx_mat) {
        Mat newBox = box.clone();
        double x0=box.get(0, 0)[0], x1=box.get(1, 0)[0], x2=box.get(2, 0)[0], x3=box.get(3, 0)[0];
        double y0=box.get(0, 1)[0], y1=box.get(1, 1)[0], y2=box.get(2, 1)[0], y3=box.get(3, 1)[0];
        double w = Math.sqrt( (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) );
        double h = Math.sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
        double box_ratio = Math.max(w, h) / (Math.min(w, h) + 1e5);

        nonzero_idx_mat = nonzero_idx_mat.reshape(1, nonzero_idx_mat.rows());
        // align diamond-shape
        if(Math.abs(1.0-box_ratio) <= 0.1) {
            Rect roi_1 = new Rect(0, 0, 1, nonzero_idx_mat.rows());
            Mat submat_1 = nonzero_idx_mat.submat(roi_1);

            Rect roi_2 = new Rect(1, 0, 1, nonzero_idx_mat.rows());
            Mat submat_2 = nonzero_idx_mat.submat(roi_2);
            MinMaxLocResult minMaxLocResult_1 = Core.minMaxLoc(submat_1);
            MinMaxLocResult minMaxLocResult_2 = Core.minMaxLoc(submat_2);
            double l = minMaxLocResult_1.minVal;
            double r = minMaxLocResult_1.maxVal;
            double t = minMaxLocResult_2.minVal;
            double b = minMaxLocResult_2.maxVal;
            Log.w("wen",  "" + l +","+ r +","+ t +","+ b);
            newBox.put(0, 0, l);
            newBox.put(0, 1, t);
            newBox.put(1, 0, r);
            newBox.put(1, 1, t);
            newBox.put(2, 0, r);
            newBox.put(2, 1, b);
            newBox.put(3, 0, l);
            newBox.put(3, 1, b);
        }

        // make clock-wise order
        double min_sum = Double.MAX_VALUE;
        int start_idx = 0;
        for(int row=0; row<4; row++) {
            double point_sums = newBox.get(row, 0)[0] + newBox.get(row, 1)[0];
            if(point_sums < min_sum) {
                min_sum = point_sums;
                start_idx = row;
            }
        }
        Mat resultBox = newBox.clone();
        for(int row=start_idx; row<start_idx+4; row++) {
            int i = row-start_idx;
            resultBox.put(i, 0, newBox.get( row%4, 0)[0]);
            resultBox.put(i, 1, newBox.get( row%4, 1)[0]);
        }
        return resultBox;
    }
    public List<Mat> adjustResultCoordinates(List<Mat> boxes, double ratio_w, double ratio_h, double ratio_net) {
        List<Mat> newBoxes = new ArrayList<Mat>();
        double x_ratio = ratio_w * ratio_net;
        double y_ratio = ratio_h * ratio_net;
        for(int b=0; b<boxes.size(); b++) {
            Mat newBox = boxes.get(b).clone();
            Rect x_roi = new Rect(0, 0, 1, 4);
            Rect y_roi = new Rect(1, 0, 1, 4);
            Mat newBox_xs = newBox.submat(x_roi);
            Mat newBox_ys = newBox.submat(y_roi);
            Core.multiply(newBox_xs, new Scalar(x_ratio), newBox_xs);
            Core.multiply(newBox_ys, new Scalar(y_ratio), newBox_ys);
            newBoxes.add(newBox);
        }
        return newBoxes;
    }

    /* --- Core Function ---*/
    public DetBoxCoreValues getDetBoxes_core(Mat score_text, Mat score_link) {
        // Threshold
        Mat threshold_score_text = score_text.clone();
        Mat threshold_score_link = score_link.clone();
        Imgproc.threshold(score_text, threshold_score_text, low_text, 1, 0);
        Imgproc.threshold(score_link, threshold_score_link, low_link, 1, 0);
        if(isDebug) MyUtils.PrintMatStat(threshold_score_text, "[Thresh_Score_text]", LogTag);
        if(isDebug) MyUtils.PrintMatStat(threshold_score_link, "[Thresh_Score_link]", LogTag);

        Mat threshold_combine = new Mat();
        Core.addWeighted(threshold_score_text, 1, threshold_score_link, 1, 0, threshold_combine);
        Imgproc.threshold(threshold_combine, threshold_combine, 0.9999, 1, 0);
        if(isDebug) MyUtils.PrintMatStat(threshold_combine, "[Thresh_Combine]", "[Text Detection]");
        threshold_combine.convertTo(threshold_combine, CvType.CV_8U);

        // Connected Components
        Mat stats = Mat.zeros(new Size(0, 0), 0);
        Mat centroids = Mat.zeros(new Size(0, 0), 0);
        Mat labels = new Mat(threshold_combine.size(), threshold_combine.type());
        Imgproc.connectedComponentsWithStats(threshold_combine, labels, stats, centroids, 4);

        List<Mat> det = new ArrayList<Mat>();
        List<Integer> mapper = new ArrayList<Integer>();

        int nLabels = centroids.rows();
        for(int k=1; k<nLabels; k++) {
            int size = (int)stats.get(k,  Imgproc.CC_STAT_AREA)[0];
            if(size < 10) continue;
            if(TextScoreMapLessThanThreshold(score_text, labels, k, text_threshold)) continue;

            Mat segmap = CreateSegMap(score_text,  labels, k);
            if(isDebug) MyUtils.PrintMatStat(segmap, "[Detected Segmap]", LogTag);
            segmap = RemoveLinkArea(segmap, threshold_score_text, threshold_score_link);
            if(isDebug) MyUtils.PrintMatStat(segmap, "[Segmap(remove link)]", LogTag);

            // Bounding Box
            int x, y, w, h, sx, ex, sy, ey;
            x = (int) stats.get(k, Imgproc.CC_STAT_LEFT)[0];
            y = (int) stats.get(k, Imgproc.CC_STAT_TOP)[0];
            w = (int) stats.get(k, Imgproc.CC_STAT_WIDTH)[0];
            h = (int) stats.get(k, Imgproc.CC_STAT_HEIGHT)[0];
            int niter = (int) ( Math.sqrt( (double) (size* Math.min(w, h)/(double)(w*h)) ) * 2.0 );
            sx = x - niter;
            ex = x + w + niter + 1;
            sy = y - niter;
            ey = y + h + niter + 1;

            // boundary check
            int img_w = segmap.cols();
            int img_h = segmap.rows();
            sx = Math.max(0, sx);
            sy = Math.max(0, sy);
            ex = Math.min(img_w, ex);
            ey = Math.min(img_h, ey);

            // dilate
            segmap = DilateSegMap(segmap, niter, sx, ex, sy, ey);
            if(isDebug) MyUtils.PrintMatStat(segmap, "[Segmap(dilated)]", LogTag);

            Mat nonzero_idx_mat = new Mat();
            Core.findNonZero(segmap, nonzero_idx_mat);
            if(isDebug) MyUtils.PrintMatStat(nonzero_idx_mat, "[NonZero Mat]", LogTag);
            MatOfPoint mop = new MatOfPoint(nonzero_idx_mat);
            MatOfPoint2f nonzero_idx_points = new MatOfPoint2f();
            mop.convertTo(nonzero_idx_points, CvType.CV_32F);
            RotatedRect rectangle = Imgproc.minAreaRect(nonzero_idx_points);
            Mat box = new Mat();
            Imgproc.boxPoints(rectangle, box);
            if(isDebug) MyUtils.PrintMatStat(box, "[Result Box]", "[Text Detection]");
            box = AlignBox(box, nonzero_idx_mat);
            det.add(box);
            mapper.add(k);
        }

        // Return Value
        DetBoxCoreValues result = new DetBoxCoreValues(det, labels, mapper);
        return result;
    }
    public List<Mat> GetBoxes(Mat img) {
        if (isDebug) MyUtils.PrintFormat(LogTag);

        // 1. Resize
        Mat img_resize = MyUtils.ResizeImg(img, target_size);
        if(isDebug) MyUtils.PrintMatStat(img_resize, "[Input Image]", LogTag);
        double ratio_w = (double)img.rows() / (double)img_resize.rows();
        double ratio_h = (double)img.cols() / (double)img_resize.cols();

        // 2. Convert Resized Mat to Bitmap
        Bitmap bitmap;
        bitmap = Bitmap.createBitmap(img_resize.cols(), img_resize.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img_resize, bitmap);

        // 3. Prepare Input Tensor
        float[] meanRGB = {0.485f, 0.456f, 0.406f};
        float[] stdRGB  = {0.229f, 0.224f, 0.225f};
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, meanRGB, stdRGB);

        // 4. Feed Into Model
        if(isDebug) MyUtils.PrintTensorStat(inputTensor, "[Input Tensor]", LogTag);
        Tensor outputTensor = this.module.forward(IValue.from(inputTensor)).toTensor();
        if(isDebug) MyUtils.PrintTensorStat(outputTensor, "[Output Tensor]", LogTag);

        // 5. Convert Tensor to Mat
        Mat score_text = OutputTensorToScoreTextMat(outputTensor);
        Mat score_link = OutputTensorToScoreLinkMat(outputTensor);

        // 6. Get Bounding Box
        DetBoxCoreValues result = getDetBoxes_core(score_text, score_link);
        List<Mat> det = result.det;
        Mat labels = result.labels;
        List<Integer> mapper = result.mapper;
        List<Mat> boxes = adjustResultCoordinates(det, ratio_w, ratio_h, 2);

        return boxes;
    }
}
