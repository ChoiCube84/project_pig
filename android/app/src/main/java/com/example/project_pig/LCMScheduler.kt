package com.example.project_pig

import java.util.Random
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Minimal LCM (Latent Consistency Model) scheduler for SD1.5-style UNet exports.
 *
 * This implementation matches diffusers' `LCMScheduler` behavior (diffusers 0.35.x):
 * - Timesteps are a subset of the distillation schedule (original_inference_steps=50 by default).
 * - Boundary-condition scaling uses `timestep_scaling` (default 10.0).
 * - Step computes x0 from epsilon prediction and injects noise except for the final step.
 *
 * Notes:
 * - Assumes UNet predicts epsilon ("prediction_type" = "epsilon").
 */
class LCMScheduler(
    private val numTrainTimesteps: Int = 1000,
    private val betaStart: Float = 0.00085f,
    private val betaEnd: Float = 0.012f,
    private val originalInferenceSteps: Int = 50,
    private val timestepScaling: Float = 10.0f,
    private val sigmaData: Float = 0.5f,
) {
    private val alphasCumprod: FloatArray
    private val finalAlphaCumprod: Float = 1.0f

    private var timesteps: IntArray = IntArray(0)

    init {
        val betas = buildScaledLinearBetas(numTrainTimesteps, betaStart, betaEnd)
        alphasCumprod = computeAlphaCumprod(betas)
    }

    fun setTimesteps(numInferenceSteps: Int, strength: Float = 1.0f) {
        require(numInferenceSteps > 0) { "numInferenceSteps must be > 0" }
        require(originalInferenceSteps in 1..numTrainTimesteps) {
            "originalInferenceSteps must be within 1..$numTrainTimesteps"
        }

        // 1) Build the LCM training/distillation timestep schedule.
        // k = num_train_timesteps // original_inference_steps
        val k = numTrainTimesteps / originalInferenceSteps
        val trainSteps = (originalInferenceSteps * strength).toInt().coerceAtLeast(1).coerceAtMost(originalInferenceSteps)
        val lcmOrigin = IntArray(trainSteps) { i ->
            ((i + 1) * k - 1).coerceIn(0, numTrainTimesteps - 1)
        }

        // 2) Reverse and pick a subset of indices evenly spaced over lcmOrigin.
        val reversed = lcmOrigin.reversedArray()
        require(numInferenceSteps <= reversed.size) {
            "numInferenceSteps ($numInferenceSteps) must be <= original schedule size (${reversed.size})"
        }

        val ts = IntArray(numInferenceSteps)
        for (i in 0 until numInferenceSteps) {
            val idx = ((i.toFloat() * reversed.size.toFloat()) / numInferenceSteps.toFloat()).toInt()
                .coerceIn(0, reversed.size - 1)
            ts[i] = reversed[idx]
        }
        timesteps = ts
    }

    fun getTimesteps(): IntArray = timesteps

    fun step(
        modelOutputEps: FloatArray,
        stepIndex: Int,
        timestep: Int,
        sample: FloatArray,
        rng: Random,
    ): FloatArray {
        val prevTimestep = if (stepIndex + 1 < timesteps.size) timesteps[stepIndex + 1] else timestep

        val alphaProdT = alphasCumprod[timestep]
        val alphaProdPrev = if (prevTimestep >= 0) alphasCumprod[prevTimestep] else finalAlphaCumprod
        val betaProdT = 1f - alphaProdT
        val betaProdPrev = 1f - alphaProdPrev

        // Boundary condition scalings (discrete)
        val scaledT = timestep.toFloat() * timestepScaling
        val denom = sqrt((scaledT * scaledT) + (sigmaData * sigmaData))
        val cSkip = (sigmaData * sigmaData) / ((scaledT * scaledT) + (sigmaData * sigmaData))
        val cOut = scaledT / denom

        val sqrtAlphaT = sqrt(alphaProdT.coerceIn(1e-8f, 1f))
        val sqrtBetaT = sqrt(betaProdT.coerceIn(0f, 1f))

        // x0 = (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)
        val predictedOriginal = FloatArray(sample.size)
        for (i in sample.indices) {
            predictedOriginal[i] = (sample[i] - (sqrtBetaT * modelOutputEps[i])) / sqrtAlphaT
        }

        // denoised = c_out * x0 + c_skip * x_t
        val denoised = FloatArray(sample.size)
        for (i in sample.indices) denoised[i] = cOut * predictedOriginal[i] + cSkip * sample[i]

        // Noise injection for multistep inference (skip on last step)
        if (stepIndex == timesteps.size - 1) return denoised

        val sqrtAlphaPrev = sqrt(alphaProdPrev.coerceIn(1e-8f, 1f))
        val sqrtBetaPrev = sqrt(betaProdPrev.coerceIn(0f, 1f))
        val noise = gaussian(rng, sample.size)

        val prev = FloatArray(sample.size)
        for (i in sample.indices) {
            prev[i] = (sqrtAlphaPrev * denoised[i]) + (sqrtBetaPrev * noise[i])
        }
        return prev
    }

    private fun buildScaledLinearBetas(numSteps: Int, betaStart: Float, betaEnd: Float): FloatArray {
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
        val alphaCp = FloatArray(betas.size)
        for (i in betas.indices) {
            alphas[i] = 1f - betas[i]
            alphaCp[i] = if (i == 0) alphas[i] else alphaCp[i - 1] * alphas[i]
        }
        return alphaCp
    }

    private fun gaussian(rnd: Random, size: Int): FloatArray {
        // Box-Muller transform.
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
