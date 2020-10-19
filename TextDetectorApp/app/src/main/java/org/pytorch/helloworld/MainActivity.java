package org.pytorch.helloworld;

import org.pytorch.helloworld.TextDetector.*;
import org.pytorch.helloworld.MyUtils.*;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.security.Permission;
import java.util.Random;

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

import android.content.Intent;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.icu.util.Output;
import android.os.Bundle;
import android.os.Debug;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.os.Looper;

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
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import com.google.android.material.snackbar.Snackbar;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.content.ContextWrapper;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;

import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.provider.MediaStore;
import android.provider.MediaStore.Images;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.database.Cursor;

import com.nabinbhandari.android.permissions.PermissionHandler;
import com.nabinbhandari.android.permissions.Permissions;

public class MainActivity extends AppCompatActivity implements View.OnClickListener{

  boolean isDebug = true;
  boolean isBilateralFilter = true;

  // DL Model
  private TextDetector textDetector;
  private TextRecognizer textRecognizer;

  // Images
  String default_image_path = "images.png";
  private enum CurDisplayImg {Ori_Img, Box_Img, BoxText_Img}
  private Mat Ori_img, Box_img, BoxText_img;
  private CurDisplayImg curDisplayImg = CurDisplayImg.Ori_Img;
  private static int RESULT_LOAD_IMAGE = 1;

  // Prediction Text
  private List<String> Predictions;

  // Buttons
  private Button btn_Ori, btn_Box, btn_BoxText;
  private Button btn_save, btn_load;

  // Request Permission
  View layout;
  final int REQUEST_READ_EXTERNAL_STORAGE = 0;
  final int REQUEST_WRITE_EXTERNAL_STORAGE = 1;
  final int REQUEST_CAMERA = 2;

  // Load OpenCV
  static {
    if(!OpenCVLoader.initDebug()){
      Log.w("[OpenCV]", "OpenCV not loaded");
    } else {
      Log.w("[OpenCV]", "OpenCV loaded");
    }
  }

  /* --- I/O Functions --- */
  public Mat GetCurImg() {
    if(curDisplayImg == CurDisplayImg.Ori_Img) return Ori_img;
    if(curDisplayImg == CurDisplayImg.Box_Img) return Box_img;
    if(curDisplayImg == CurDisplayImg.BoxText_Img) return BoxText_img;
    else return null;
  }
  public Bitmap readBitmap(String fileName) {
    BitmapFactory.Options op = new BitmapFactory.Options();
    op.inPreferredConfig = Bitmap.Config.ARGB_8888;

    try {
      //if(isDebug) Log.w("wen", "Before Read Image");
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(fileName));
      //if(isDebug) Log.w("wen", "After Read Image");
      return bitmap;
    }
    catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading imgs");
      finish();
      return null;
    }
  }
  public Module loadModule(String fileName) {
    try {
      //if(isDebug) Log.w("wen", "Before Load Model");
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      Module module = Module.load(assetFilePath(this, fileName));
      //if(isDebug) Log.w("wen", "After Load Model");
      return module;
    }
    catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error Load Model");
      finish();
      return null;
    }
  }
  public Mat readImgMat(String image_path) {

    // Read Bitmap
    Bitmap bitmap = readBitmap(image_path);

    // Convert Bitmap to Mat
    Mat img = new Mat();
    Utils.bitmapToMat(bitmap, img);
    return img;
  }
  public void DisplayMat(Mat img) {
    // Create Bitmap for display
    Bitmap bitmap_display = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);

    // Convert Mat to Bitmap
    Utils.matToBitmap(img, bitmap_display);

    // Get Image View
    ImageView imageView = findViewById(R.id.image);

    // Display
    int screen_height = getResources().getDisplayMetrics().heightPixels;
    imageView.getLayoutParams().height = (int) (screen_height * 0.75);
    imageView.setImageBitmap(bitmap_display);


  }
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  // Save To Gallery
  private void galleryAddPic(String interalPath) {
    Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
    File f = new File(interalPath);
    Uri contentUri = Uri.fromFile(f);
    mediaScanIntent.setData(contentUri);
    this.sendBroadcast(mediaScanIntent);
  }
  private String saveToInternalStorage(Mat img, String save_name){
    Bitmap bitmapImage = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(img, bitmapImage);

    MediaStore.Images.Media.insertImage(getContentResolver(), bitmapImage, "a.jpg" , "Test");

    ContextWrapper cw = new ContextWrapper(getApplicationContext());
    // path to /data/data/yourapp/app_data/imageDir
    File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
    // Create imageDir
    File mypath=new File(directory, save_name);

    FileOutputStream fos = null;
    try {
      fos = new FileOutputStream(mypath);
      // Use the compress method on the BitMap object to write image to the OutputStream
      bitmapImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      try {
        fos.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    return mypath.getAbsolutePath();
  }

  // Load From Gallery
  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);

    if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
      Uri selectedImage = data.getData();
      String[] filePathColumn = {MediaStore.Images.Media.DATA};

      Cursor cursor = getContentResolver().query(selectedImage,
              filePathColumn, null, null, null);
      cursor.moveToFirst();

      int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
      String picturePath = cursor.getString(columnIndex);
      cursor.close();

      Bitmap bitmap = BitmapFactory.decodeFile(picturePath);
      Utils.bitmapToMat(bitmap, Ori_img);
      curDisplayImg = CurDisplayImg.Ori_Img;
      Display();

      Toast.makeText(MainActivity.this, "Load Image", Toast.LENGTH_SHORT).show();

      // Invalid Box and Text
      Box_img = null;
      BoxText_img = null;
      Predictions = null;
      Display();
    }
  }

    /* --- Buttons Functions --- */
  public void InitBtnListeners() {

    btn_Ori = (Button) findViewById(R.id.btn_ori_img) ;
    btn_Ori.setOnClickListener(this);

    btn_Box = (Button) findViewById(R.id.btn_box_img) ;
    btn_Box.setOnClickListener(this);

    btn_BoxText = (Button) findViewById(R.id.btn_box_text_img) ;
    btn_BoxText.setOnClickListener(this);

    btn_load = (Button) findViewById(R.id.btn_load);
    btn_load.setOnClickListener(this);

    btn_save = (Button) findViewById(R.id.btn_save);
    btn_save.setOnClickListener(this);
  }

  /* --- Permission Functions --- */
  void RequestWritePermission(){
    Permissions.check(this, Manifest.permission.WRITE_EXTERNAL_STORAGE, null, new PermissionHandler() {
      @Override
      public void onGranted() {
        Log.w("[Permission]", "Accept Write");
        String interalPath = saveToInternalStorage(GetCurImg(), "a.jpg");
        galleryAddPic(interalPath);
        Toast.makeText(MainActivity.this, "Save Image", Toast.LENGTH_SHORT).show();
      }
    });
  }
  void RequestReadPermission() {
    Permissions.check(this, Manifest.permission.READ_EXTERNAL_STORAGE, null, new PermissionHandler() {
      @Override
      public void onGranted() {
        Log.w("[Permission]", "Accept Read");

        Intent i = new Intent(
                Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, RESULT_LOAD_IMAGE );
      }
    });
  }

  @Override
  public void onClick(View v) {
    if (v==btn_save) {
      RequestWritePermission();
      return;
    }
    if (v==btn_load) {
      RequestReadPermission();
      return;
    }
    if (v==btn_Ori) curDisplayImg = CurDisplayImg.Ori_Img;
    if (v==btn_Box) curDisplayImg = CurDisplayImg.Box_Img;
    if (v==btn_BoxText) curDisplayImg = CurDisplayImg.BoxText_Img;
    Display();
  }
  public void Display() {
    if(GetCurImg() == null)
      ProcessImg(Ori_img);

    // Display Current Image
    DisplayMat(GetCurImg());

    // Display Prediction Text
    TextView textView = findViewById(R.id.text);
    textView.setText("Predict : " + MyUtils.GetListString(Predictions));
  }

  /* --- Core Function --- */
  public void ProcessImg(Mat img) {

    new Thread(new Runnable() {
      public void run() {
        Looper.prepare();
        Toast.makeText(MainActivity.this, "Processing Image, Please Wait..", Toast.LENGTH_LONG).show();
        Looper.loop();
      }
    }).start();

    Ori_img = MyUtils.ToColorMat(img);

    // Get Boxes
    List<Mat> boxes = textDetector.GetBoxes(img);
    Box_img = MyUtils.DrawBoxes(img, boxes);

    Predictions = new ArrayList<String>();
    String preds_cat = "";
    BoxText_img = MyUtils.ToColorMat(img);
    for(int b=0; b<boxes.size(); b++) {
      // Crop Image by box
      Mat box = boxes.get(b);
      Mat img_crop = MyUtils.CropImg(img, box);

      // Get Text Predictions
      String pred_string = textRecognizer.GetText(img_crop);

      // Draw Box and Text
      BoxText_img = MyUtils.DrawBoxText(BoxText_img, box, pred_string);

      Predictions.add(pred_string);
      preds_cat += pred_string;
    }
    Display();
  }

  /* --- Init Function --- */
  @Override
  protected void onCreate(Bundle savedInstanceState) {

    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    InitBtnListeners();

    /* ---  Load Default Image --- */
    Mat default_img = readImgMat(default_image_path);

    /* --- Load Module --- */
    Module text_detector_module   = loadModule("text_detector.pt");
    Module text_recognizer_module = loadModule("text_recognizer.pt");
    this.textDetector = new TextDetector(text_detector_module, isDebug);
    this.textRecognizer = new TextRecognizer(text_recognizer_module, isDebug, isBilateralFilter);

    /* --- Processing Default Image --- */
    Ori_img = default_img.clone();
    Display();
    // ProcessImg(default_img);
  }

}
