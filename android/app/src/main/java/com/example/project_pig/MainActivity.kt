package com.example.project_pig

import android.graphics.Bitmap
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView

class MainActivity : AppCompatActivity() {

    private lateinit var sdPipeline: SDPipeline

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val etPrompt = findViewById<EditText>(R.id.etPrompt)
        val btnGenerate = findViewById<Button>(R.id.btnGenerate)
        val tvStatus = findViewById<TextView>(R.id.tvStatus)
        val ivResult = findViewById<ImageView>(R.id.ivResult)

        // 파이프라인 초기화 (앱 시작 시 모델 로딩)
        // 실제로는 로딩 화면을 보여주는 것이 좋습니다.
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                sdPipeline = SDPipeline(applicationContext)
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Status: Model Loaded"
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        btnGenerate.setOnClickListener {
            val prompt = etPrompt.text.toString()

            // UI 블로킹 방지
            btnGenerate.isEnabled = false

            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    val resultBitmap = sdPipeline.generate(prompt) { status ->
                        // 진행 상황 업데이트 (UI 스레드)
                        runOnUiThread { tvStatus.text = status }
                    }

                    // 결과 표시
                    withContext(Dispatchers.Main) {
                        ivResult.setImageBitmap(resultBitmap)
                        btnGenerate.isEnabled = true
                        tvStatus.text = "Status: Completed"
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    runOnUiThread {
                        tvStatus.text = "Error: ${e.message}"
                        btnGenerate.isEnabled = true
                    }
                }
            }
        }
    }
}