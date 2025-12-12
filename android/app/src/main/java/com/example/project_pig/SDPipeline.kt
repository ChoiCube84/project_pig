package com.example.project_pig

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.max
import kotlin.math.sqrt

class SDPipeline(context: Context) {

    private val env = OrtEnvironment.getEnvironment()

    // 세션 변수
    private var textEncoderSession: OrtSession? = null
    private var unetSession: OrtSession? = null
    private var vaeDecoderSession: OrtSession? = null

    init {
        try {
            // 옵션 설정 (필요시 NNAPI 등 가속기 설정 가능)
            val options = OrtSession.SessionOptions()
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)

            // options.addNnapi() // When activate this, they say the binary was not quantized for NNAPI or something. I'll have to apply it later or ignore forever

            // [중요] 파일을 직접 읽지 않고, 파일 경로를 통해 로드합니다 (Memory Mapping)
            Log.d("SDPipeline", "Copying models to cache... (This runs only once)")

            val textEncoderPath = copyAssetToFile(context, "text_encoder.onnx")
            val unetPath = copyAssetToFile(context, "unet_quantized.onnx") // 파일명 확인!
            val vaePath = copyAssetToFile(context, "vae_decoder.onnx")

            Log.d("SDPipeline", "Loading ONNX Sessions...")
            textEncoderSession = env.createSession(textEncoderPath, options)
            unetSession = env.createSession(unetPath, options)
            vaeDecoderSession = env.createSession(vaePath, options)

            Log.d("SDPipeline", "Models Loaded Successfully!")

        } catch (e: Exception) {
            Log.e("SDPipeline", "Error initializing model", e)
            throw e
        }
    }

    // [핵심] Assets에 있는 파일을 내부 저장소로 복사하고 경로를 반환하는 함수
    private fun copyAssetToFile(context: Context, fileName: String): String {
        val file = File(context.filesDir, fileName)
        // 파일이 없거나 용량이 다르면 복사 (이미 있으면 스킵하여 속도 향상)
        if (!file.exists() || file.length() == 0L) {
            context.assets.open(fileName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return file.absolutePath
    }

    // === 아래는 기존과 동일 ===

    fun generate(prompt: String, callback: (String) -> Unit): Bitmap {
        callback("Encoding Text...")
        val textEmbeddings = encodeText(prompt)

        callback("Denoising...")
        // 1. 랜덤 노이즈 생성 (캔버스)
        var latents = generateRandomLatents(1, 4, 64, 64)

        // 2. 스케줄러 준비 (간단한 DDPM/선형 베타 스케줄)
        val numTrainSteps = 1000
        // CLIP/DDPM scaled_linear beta schedule (same as diffusers PNDMScheduler default)
        val betas = buildScaledLinearBetas(numTrainSteps, 0.00085f, 0.012f)
        val alphaCumprod = computeAlphaCumprod(betas)

        // 2. 확산 모델 루프
        // 높은 timestep(거의 맨 처음 노이즈)에 대해 여러 번 반복해야 색이 나옵니다.
        val steps = 30
        // leading spacing: evenly spaced indices from high -> low
        val timesteps = IntArray(steps) { idx ->
            ((numTrainSteps - 1) - idx * (numTrainSteps.toFloat() / steps)).toInt().coerceAtLeast(0)
        }

        for (i in 0 until steps) {
            callback("Step ${i+1} / $steps")

            val t = timesteps[i]
            val noisePredTensor = runUNetStep(latents, textEmbeddings, t)

            // DDPM 한 스텝: x_{t-1} = sqrt(alpha_prev)*x0 + sqrt(1-alpha_prev)*eps
            val prevT = if (i == steps - 1) 0 else timesteps[i + 1]
            val updatedLatents = ddpmStep(latents, noisePredTensor, t, prevT, alphaCumprod)

            noisePredTensor.close()
            latents.close()
            latents = updatedLatents
        }

        callback("Decoding Image...")

        val imageBitmap = decodeVAE(latents)
        latents.close()
        textEmbeddings.close()

        callback("Done!")
        return imageBitmap
    }

    // [추가] 텐서 뺄셈 함수 (Latents - Noise)
    private fun ddpmStep(latents: OnnxTensor, noise: OnnxTensor, t: Int, prevT: Int, alphaCumprod: FloatArray): OnnxTensor {
        val latArray = tensorToArray(latents)
        val noiseArray = tensorToArray(noise)

        val alphaT = alphaCumprod[t]
        val alphaPrev = alphaCumprod[prevT]

        val sqrtAlphaT = sqrt(alphaT)
        val sqrtOneMinusAlphaT = sqrt(1f - alphaT)
        val sqrtAlphaPrev = sqrt(alphaPrev)
        val sqrtOneMinusAlphaPrev = sqrt(1f - alphaPrev)

        val size = latArray.size
        val out = FloatArray(size)

        // x0 추정 후 t-1 스텝 계산
        for (i in 0 until size) {
            val xT = latArray[i]
            val eps = noiseArray[i]
            val x0 = (xT - sqrtOneMinusAlphaT * eps) / sqrtAlphaT
            out[i] = (sqrtAlphaPrev * x0) + (sqrtOneMinusAlphaPrev * eps)
        }

        return OnnxTensor.createTensor(env, FloatBuffer.wrap(out), longArrayOf(1, 4, 64, 64))
    }

    private fun tensorToArray(tensor: OnnxTensor): FloatArray {
        val buffer = tensor.floatBuffer
        buffer.rewind()
        val arr = FloatArray(buffer.capacity())
        buffer.get(arr)
        buffer.rewind()
        return arr
    }

    private fun buildLinearBetas(numSteps: Int, betaStart: Float, betaEnd: Float): FloatArray {
        val betas = FloatArray(numSteps)
        val delta = (betaEnd - betaStart) / (numSteps - 1)
        for (i in 0 until numSteps) {
            betas[i] = betaStart + delta * i
        }
        return betas
    }

    private fun buildScaledLinearBetas(numSteps: Int, betaStart: Float, betaEnd: Float): FloatArray {
        // matches diffusers scaled_linear schedule: linspace(sqrt(start), sqrt(end))^2
        val betas = FloatArray(numSteps)
        val startS = sqrt(betaStart)
        val endS = sqrt(betaEnd)
        val delta = (endS - startS) / (numSteps - 1)
        for (i in 0 until numSteps) {
            val valS = startS + delta * i
            betas[i] = valS * valS
        }
        return betas
    }

    private fun computeAlphaCumprod(betas: FloatArray): FloatArray {
        val alphas = FloatArray(betas.size)
        val alphaCumprod = FloatArray(betas.size)
        for (i in betas.indices) {
            alphas[i] = 1f - betas[i]
            if (i == 0) {
                alphaCumprod[i] = alphas[i]
            } else {
                alphaCumprod[i] = alphaCumprod[i - 1] * alphas[i]
            }
        }
        return alphaCumprod
    }

    private fun encodeText(prompt: String): OnnxTensor {
        val seqLength = 77
        val longArray = LongArray(seqLength) { 0L }
        longArray[0] = 49407L

        val byteBuffer = ByteBuffer.allocateDirect(seqLength * 8)
            .order(ByteOrder.nativeOrder())
        val longBuffer = byteBuffer.asLongBuffer()
        longBuffer.put(longArray)
        longBuffer.rewind()

        val shape = longArrayOf(1, seqLength.toLong())
        val inputTensor = OnnxTensor.createTensor(env, longBuffer, shape)

        val result = textEncoderSession!!.run(Collections.singletonMap("input_ids", inputTensor))
        inputTensor.close()

        // Clone out of the Result so we can close it
        val embeddings = result[0] as OnnxTensor
        // DO NOT close embeddings here, caller will close it
        result.close()

        return embeddings
    }
    private fun runUNetStep(latents: OnnxTensor, encoderHiddenStates: OnnxTensor, timestep: Int): OnnxTensor {
        val timestepTensor = OnnxTensor.createTensor(env, floatArrayOf(timestep.toFloat()))
        val inputs = mapOf(
            "sample" to latents,
            "timestep" to timestepTensor,
            "encoder_hidden_states" to encoderHiddenStates
        )

        val result = unetSession!!.run(inputs)
        timestepTensor.close()

        val noisePred = result[0] as OnnxTensor
        result.close()
        return noisePred
    }

    private fun decodeVAE(latents: OnnxTensor): Bitmap {
        Log.d("SDPipeline", "Scaling latents before VAE...")

        val scalingFactor = 1.0f / 0.18215f
        val originalBuffer = latents.floatBuffer
        originalBuffer.rewind()
        val size = originalBuffer.capacity()

        val scaledBuffer = FloatBuffer.allocate(size)
        for (i in 0 until size) {
            val value = originalBuffer.get(i)
            scaledBuffer.put(value * scalingFactor)
        }
        scaledBuffer.rewind()

        val scaledTensor = OnnxTensor.createTensor(env, scaledBuffer, longArrayOf(1, 4, 64, 64))
        val inputs = mapOf("latent_sample" to scaledTensor)

        val result = vaeDecoderSession!!.run(inputs)
        scaledTensor.close()

        val imageTensor = result[0] as OnnxTensor
        result.close()

        val bitmap = postProcessImage(imageTensor)
        imageTensor.close()
        return bitmap
    }

    private fun generateRandomLatents(b: Int, c: Int, h: Int, w: Int): OnnxTensor {
        val buffer = FloatBuffer.allocate(b * c * h * w)
        // 0.0f로 채우면 재미 없으니 랜덤값 살짝 넣기 (실제로는 Gaussian Noise 필요)
        for(i in 0 until buffer.capacity()) buffer.put((Math.random().toFloat() - 0.5f))
        buffer.rewind()
        return OnnxTensor.createTensor(env, buffer, longArrayOf(b.toLong(), c.toLong(), h.toLong(), w.toLong()))
    }

    private fun postProcessImage(tensor: OnnxTensor): Bitmap {
        // 1. 텐서에서 데이터 꺼내기 (FloatBuffer)
        val floatBuffer = tensor.floatBuffer

        val width = 512
        val height = 512
        val channelSize = width * height // 512*512 = 262144

        // 픽셀을 담을 배열 (ARGB 포맷)
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                // ONNX 결과는 보통 Planar 포맷 (RRR...GGG...BBB...)
                // 배열 인덱스 계산
                val index = y * width + x
                val rIdx = 0 * channelSize + index
                val gIdx = 1 * channelSize + index
                val bIdx = 2 * channelSize + index

                // 값 읽기 (VAE 출력 범위는 보통 -1.0 ~ 1.0)
                // 만약 범위가 0~1이라면 /2 + 0.5 연산은 빼야 합니다. (보통은 -1~1임)
                val rRaw = floatBuffer.get(rIdx)
                val gRaw = floatBuffer.get(gIdx)
                val bRaw = floatBuffer.get(bIdx)

                // -1.0 ~ 1.0 -> 0 ~ 255 변환
                val r = ((rRaw / 2.0f + 0.5f).coerceIn(0f, 1f) * 255f).toInt()
                val g = ((gRaw / 2.0f + 0.5f).coerceIn(0f, 1f) * 255f).toInt()
                val b = ((bRaw / 2.0f + 0.5f).coerceIn(0f, 1f) * 255f).toInt()

                // ARGB Int로 패킹 (Alpha는 255 완전 불투명)
                val color = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                pixels[index] = color
            }
        }

        // 비트맵 생성 및 픽셀 입력
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
