package com.example.project_pig

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.content.ContentValues
import android.provider.MediaStore
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    // 초기화가 나중에 된다는 뜻 (lateinit)
    private lateinit var sdPipeline: SDPipeline
    private var isModelLoaded = false
    private var lastBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val etPrompt = findViewById<EditText>(R.id.etPrompt)
        val etSeed = findViewById<EditText>(R.id.etSeed)
        val btnGenerate = findViewById<Button>(R.id.btnGenerate)
        val btnSave = findViewById<Button>(R.id.btnSave)
        val tvStatus = findViewById<TextView>(R.id.tvStatus)
        val ivResult = findViewById<ImageView>(R.id.ivResult)

        // 1. 앱 켜자마자 버튼 비활성화 (못 누르게 막음)
        btnGenerate.isEnabled = false
        btnSave.isEnabled = false
        tvStatus.text = "Initializing Model... (This may take a few seconds)"

        // 2. 백그라운드에서 모델 로딩 시작
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d("MainActivity", "Start loading pipeline...")

                // 여기서 시간이 오래 걸립니다 (파일 복사 + ONNX 로드)
                val pipeline = SDPipeline(applicationContext)

                // 로딩 성공 시
                withContext(Dispatchers.Main) {
                    sdPipeline = pipeline
                    isModelLoaded = true
                    btnGenerate.isEnabled = true // 이제 버튼 활성화!
                    tvStatus.text = "Model Loaded! Ready to generate."
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Model Load Failed: ${e.message}"
                }
            }
        }

        // 3. 버튼 클릭 이벤트
        btnGenerate.setOnClickListener {
            if (!isModelLoaded) return@setOnClickListener

            val prompt = etPrompt.text.toString()
            val seed = etSeed.text.toString().toLongOrNull() ?: 42L
            btnGenerate.isEnabled = false // 중복 클릭 방지
            btnSave.isEnabled = false
            lastBitmap = null

            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    val resultBitmap = sdPipeline.generate(
                        prompt = prompt,
                        seed = seed,
                        callback = { status -> runOnUiThread { tvStatus.text = status } },
                    )

                    withContext(Dispatchers.Main) {
                        lastBitmap = resultBitmap
                        ivResult.setImageBitmap(resultBitmap)
                        btnGenerate.isEnabled = true
                        btnSave.isEnabled = true
                        tvStatus.text = "Done!"
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    withContext(Dispatchers.Main) {
                        tvStatus.text = "Error: ${e.message}"
                        btnGenerate.isEnabled = true
                    }
                }
            }
        }

        btnSave.setOnClickListener {
            val bitmap = lastBitmap ?: return@setOnClickListener
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val uri = saveBitmapToGallery(bitmap)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Saved: $uri", Toast.LENGTH_LONG).show()
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "Failed to save image", e)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap): String {
        val resolver = applicationContext.contentResolver
        val fileName = "project_pig_${System.currentTimeMillis()}.png"

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ProjectPig")
        }

        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("MediaStore insert failed")

        resolver.openOutputStream(uri)?.use { out ->
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                error("Bitmap compress failed")
            }
        } ?: error("Failed to open output stream")

        return uri.toString()
    }
}
