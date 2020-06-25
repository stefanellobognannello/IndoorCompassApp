package org.pytorch.IndoorCompass;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
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
import java.util.Arrays;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;


public class MainActivity extends AppCompatActivity {
  int cameraRequestCode = 001;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Button capture = findViewById(R.id.button2);

    capture.setOnClickListener(new View.OnClickListener() {

      @Override
      public void onClick(View view) {

        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        startActivityForResult(cameraIntent, cameraRequestCode);

      }


    });

  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {

    super.onActivityResult(requestCode, resultCode, data);
    super.onActivityResult(requestCode, resultCode, data);
    if (requestCode == cameraRequestCode && resultCode == RESULT_OK) {

      Intent resultView = new Intent(this, MainActivity.class);

      resultView.putExtra("imagedata", data.getExtras());


      Bitmap bitmap = null;
      Module module = null;
      try {
        // creating bitmap from packaged into app android asset 'image.jpg',
        // app/src/main/assets/image.jpg
        bitmap = (Bitmap) data.getExtras().get("data");

        // loading serialized torchscript module from packaged into app android asset model.pt,
        // app/src/model/assets/model.pt
        module = Module.load(assetFilePath(this, "resnet18_finetuning_traced_2fps_200_4.pt"));
      } catch (IOException e) {
        Log.e("IndoorCompass", "Error reading assets", e);
        finish();
      }

      // showing image on UI
      bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
      ImageView imageView = findViewById(R.id.image);
      imageView.setImageBitmap(bitmap);

      // preparing input tensor

      float mean[] = new float[]{0.33210807f, 0.29329791f, 0.26337797f};
      float std[] = new float[]{0.24047631f, 0.23936823f, 0.24046722f};
      final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, mean, std);

      // running the model
      final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

      // getting tensor content as java array of floats
      final float[] scores = outputTensor.getDataAsFloatArray();

      // searching for the index with maximum score
      float maxScore = -Float.MAX_VALUE;
      int maxScoreIdx = -1;
      int midScoreIdx;
      int lowScoreIdx;
      for (int i = 0; i < scores.length; i++) {
        if (scores[i] > maxScore) {
          maxScore = scores[i];
          maxScoreIdx = i;
        }
      }

      if (scores[(maxScoreIdx + 1) % 3] > scores[(maxScoreIdx + 2) % 3]) {
        midScoreIdx = (maxScoreIdx + 1) % 3;
        lowScoreIdx = (maxScoreIdx + 2) % 3;
      } else {
        lowScoreIdx = (maxScoreIdx + 1) % 3;
        midScoreIdx = (maxScoreIdx + 2) % 3;
      }

      String className2 = IndoorCompassClasses.INDOORCOMPASS_CLASSES[maxScoreIdx];
      String className1 = IndoorCompassClasses.INDOORCOMPASS_CLASSES[midScoreIdx];
      String className0 = IndoorCompassClasses.INDOORCOMPASS_CLASSES[lowScoreIdx];

      //softmax on values
      double[] dscores=convertFloatsToDoubles(scores);
      double total = Arrays.stream(dscores).map(Math::exp).sum();
      double val2 = Math.exp(scores[maxScoreIdx]) / total;
      double val1 = Math.exp(scores[midScoreIdx]) / total;
      double val0 = Math.exp(scores[lowScoreIdx]) / total;
      // showing className on UI
      TextView textView2 = findViewById(R.id.text2);
      textView2.setText(className2 +" "+round(val2*100,2)+"%");

      TextView textView1 = findViewById(R.id.text1);
      textView1.setText(className1+" "+round(val1*100,2)+"%");

      TextView textView0 = findViewById(R.id.text0);
      textView0.setText(className0+" "+round(val0*100,2)+"%");


    }

  }


  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
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

  public static double[] convertFloatsToDoubles(float[] input) {
    if (input == null) {
      return null; // Or throw an exception - your choice
    }
    double[] output = new double[input.length];
    for (int i = 0; i < input.length; i++) {
      output[i] = input[i];
    }
    return output;
  }

  public static double round(double value, int places) {
    if (places < 0) throw new IllegalArgumentException();

    long factor = (long) Math.pow(10, places);
    value = value * factor;
    long tmp = Math.round(value);
    return (double) tmp / factor;
  }
}
