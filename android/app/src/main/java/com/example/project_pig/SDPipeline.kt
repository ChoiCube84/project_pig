package com.example.project_pig

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Collections
import java.util.Random
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.math.sin

// Shared resources to survive Activity recreation (e.g., rotation) and avoid reloading models.
private const val GUIDANCE_SCALE = 7.5f

class SDPipeline(context: Context) {

    companion object {
        private val lock = Any()
        private val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
        private var sharedTokenizer: ClipTokenizer? = null
        private var sharedTextEncoderSession: OrtSession? = null
        private var sharedUnetSession: OrtSession? = null
        private var sharedVaeSession: OrtSession? = null
    }

    // 세션 변수
    private var textEncoderSession: OrtSession? = null
    private var unetSession: OrtSession? = null
    private var vaeDecoderSession: OrtSession? = null
    private var tokenizer: ClipTokenizer

    init {
        val appCtx = context.applicationContext
        synchronized(lock) {
            try {
                if (sharedTokenizer == null) {
                    sharedTokenizer = ClipTokenizer(appCtx)
                }
                tokenizer = sharedTokenizer!!

                if (sharedTextEncoderSession == null || sharedUnetSession == null || sharedVaeSession == null) {
                    val options = OrtSession.SessionOptions()
                    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)

                    // options.addNnapi()

                    Log.d("SDPipeline", "Copying models to cache... (first run only)")
                    val textEncoderPath = copyAssetToFile(appCtx, "text_encoder.onnx")
                    val unetPath = copyAssetToFile(appCtx, "unet_quantized.onnx")
                    val vaePath = copyAssetToFile(appCtx, "vae_decoder.onnx")

                    Log.d("SDPipeline", "Loading ONNX Sessions...")
                    sharedTextEncoderSession = env.createSession(textEncoderPath, options)
                    sharedUnetSession = env.createSession(unetPath, options)
                    sharedVaeSession = env.createSession(vaePath, options)
                    Log.d("SDPipeline", "Models Loaded Successfully!")
                }

                textEncoderSession = sharedTextEncoderSession
                unetSession = sharedUnetSession
                vaeDecoderSession = sharedVaeSession
            } catch (e: Exception) {
                Log.e("SDPipeline", "Error initializing model", e)
                throw e
            }
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
        // unconditional (empty) and conditional embeddings for classifier-free guidance
        val uncondEmb = encodeText("")
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
        // Match diffusers: linspace(0, numTrainSteps-1, steps)[::-1] so we reach t=0.
        val timesteps = IntArray(steps) { idx ->
            val t = ((steps - 1 - idx).toLong() * (numTrainSteps - 1).toLong()) / (steps - 1).toLong()
            t.toInt().coerceIn(0, numTrainSteps - 1)
        }

        for (i in 0 until steps) {
            callback("Step ${i+1} / $steps")

            val t = timesteps[i]
            val prevT = if (i == steps - 1) -1 else timesteps[i + 1]

            // classifier-free guidance: eps = eps_uncond + scale*(eps_text - eps_uncond)
            val epsUncond = runUNetStep(latents, uncondEmb, t)
            val epsText = runUNetStep(latents, textEmbeddings, t)
            val guidedEps = combineGuidance(epsUncond, epsText, guidanceScale = GUIDANCE_SCALE)

            // DDIM step (eta=0 deterministic)
            val updatedLatents = ddimStep(latents, guidedEps, t, prevT, alphaCumprod, eta = 0f)

            epsUncond.close()
            epsText.close()
            guidedEps.close()
            latents.close()
            latents = updatedLatents
        }

        callback("Decoding Image...")

        val imageBitmap = decodeVAE(latents)
        latents.close()
        textEmbeddings.close()
        uncondEmb.close()

        callback("Done!")
        return imageBitmap
    }

    // [추가] 텐서 뺄셈 함수 (Latents - Noise)
    private fun ddimStep(latents: OnnxTensor, eps: OnnxTensor, t: Int, prevT: Int, alphaCumprod: FloatArray, eta: Float): OnnxTensor {
        val latArray = tensorToArray(latents)
        val epsArray = tensorToArray(eps)

        val alphaT = alphaCumprod[t]
        val alphaPrev = if (prevT < 0) 1f else alphaCumprod[prevT]

        val sqrtAlphaT = sqrt(alphaT)
        val sqrtOneMinusAlphaT = sqrt(1f - alphaT)
        val sqrtAlphaPrev = sqrt(alphaPrev)

        val sigma = if (eta == 0f) 0f else eta * sqrt((1 - alphaPrev) / (1 - alphaT) * (1 - alphaT / alphaPrev))
        val noiseScale = sqrtOneMinusAlphaT
        val dirCoeff = sqrt((1f - alphaPrev) - (sigma * sigma)).coerceAtLeast(0f)

        val size = latArray.size
        val out = FloatArray(size)
        val noise = if (sigma == 0f) null else generateGaussian(size)

        for (i in 0 until size) {
            val xT = latArray[i]
            val e = epsArray[i]
            val predX0 = (xT - noiseScale * e) / sqrtAlphaT
            val predDir = dirCoeff * e
            val noiseTerm = if (sigma == 0f) 0f else sigma * (noise!![i])
            out[i] = (sqrtAlphaPrev * predX0) + predDir + noiseTerm
        }

        return OnnxTensor.createTensor(env, FloatBuffer.wrap(out), longArrayOf(1, 4, 64, 64))
    }

    private fun combineGuidance(uncond: OnnxTensor, cond: OnnxTensor, guidanceScale: Float): OnnxTensor {
        val u = tensorToArray(uncond)
        val c = tensorToArray(cond)
        val out = FloatArray(u.size)
        for (i in u.indices) {
            out[i] = u[i] + guidanceScale * (c[i] - u[i])
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(out), uncond.info.shape)
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
        val tokenIds = tokenizer.encode(prompt)

        val byteBuffer = ByteBuffer.allocateDirect(tokenIds.size * java.lang.Long.BYTES)
            .order(ByteOrder.nativeOrder())
        val longBuffer = byteBuffer.asLongBuffer()
        longBuffer.put(tokenIds)
        longBuffer.rewind()

        val shape = longArrayOf(1, tokenIds.size.toLong())
        val inputTensor = OnnxTensor.createTensor(env, longBuffer, shape)

        val result = textEncoderSession!!.run(Collections.singletonMap("input_ids", inputTensor))
        inputTensor.close()

        val embeddings = result[0] as OnnxTensor

        val outShape = embeddings.info.shape
        return if (outShape.size == 3) {
            embeddings
        } else {
            // Some builds may return an extra leading dim; reshape to [1, seq, 768]
            val flat = tensorToArray(embeddings)
            embeddings.close()
            val seq = tokenIds.size.toLong()
            OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), longArrayOf(1, seq, 768))
        }
    }
    private fun runUNetStep(latents: OnnxTensor, encoderHiddenStates: OnnxTensor, timestep: Int): OnnxTensor {
        // UNet expects a scalar timestep (shape []), not [1]. Use a direct buffer with empty shape.
        val tsBuffer = ByteBuffer.allocateDirect(java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        tsBuffer.put(timestep.toFloat())
        tsBuffer.rewind()
        val timestepTensor = OnnxTensor.createTensor(env, tsBuffer, longArrayOf())
        val inputs = mapOf(
            "sample" to latents,
            "timestep" to timestepTensor,
            "encoder_hidden_states" to encoderHiddenStates
        )

        val result = unetSession!!.run(inputs)
        timestepTensor.close()

        val noisePred = result[0] as OnnxTensor
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

        val bitmap = postProcessImage(imageTensor)
        imageTensor.close()
        return bitmap
    }

    private fun generateRandomLatents(b: Int, c: Int, h: Int, w: Int): OnnxTensor {
        val buffer = FloatBuffer.allocate(b * c * h * w)
        // Stable Diffusion expects standard normal noise (Gaussian), not uniform.
        val noise = generateGaussian(buffer.capacity())
        for (i in 0 until buffer.capacity()) buffer.put(noise[i])
        buffer.rewind()
        return OnnxTensor.createTensor(env, buffer, longArrayOf(b.toLong(), c.toLong(), h.toLong(), w.toLong()))
    }

    private fun postProcessImage(tensor: OnnxTensor): Bitmap {
        // 1. 텐서에서 데이터 꺼내기 (FloatBuffer)
        val floatBuffer = tensor.floatBuffer
        floatBuffer.rewind()

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

    private fun generateGaussian(size: Int, seed: Long = System.nanoTime()): FloatArray {
        // Box-Muller transform.
        val rnd = Random(seed)
        val out = FloatArray(size)
        var i = 0
        while (i < size) {
            val u1 = rnd.nextDouble().coerceAtLeast(1e-12)
            val u2 = rnd.nextDouble()
            val r = sqrt((-2.0 * ln(u1)).toFloat())
            val theta = (2.0 * Math.PI * u2).toFloat()
            out[i] = r * cos(theta)
            if (i + 1 < size) out[i + 1] = r * sin(theta)
            i += 2
        }
        return out
    }
}
