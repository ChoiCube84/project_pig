package com.example.project_pig

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.regex.Pattern

/**
 * Minimal GPT-2/CLIP byte-level BPE tokenizer for SD v1.5 text encoder.
 * Loads vocab/merges from assets (vocab.json, merges.txt) and supports BOS/EOS, padding.
 */
class ClipTokenizer(context: Context) {
    private val vocab: Map<String, Int>
    private val bpeRanks: Map<Pair<String, String>, Int>

    private val byteEncoder: Map<Int, String>
    private val byteDecoder: Map<String, Int>

    private val pattern = Pattern.compile("""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\\s]+|\\s+(?!\\S)|\\s+""")

    private val bosToken = 49406
    private val eosToken = 49407
    private val padToken = 0
    private val maxLength = 77

    init {
        vocab = loadVocab(context)
        bpeRanks = loadMerges(context)
        val enc = buildByteEncoder()
        byteEncoder = enc
        byteDecoder = enc.entries.associate { it.value to it.key }
    }

    fun encode(text: String): LongArray {
        val tokens = mutableListOf<Int>()
        tokens.add(bosToken)

        val matcher = pattern.matcher(text)
        while (matcher.find()) {
            val token = matcher.group()
            val encoded = tokenToBpeTokens(token)
            tokens.addAll(encoded)
            if (tokens.size >= maxLength - 1) break
        }

        // add EOS
        if (tokens.size < maxLength) tokens.add(eosToken)

        // pad/truncate
        if (tokens.size > maxLength) {
            return tokens.take(maxLength).map { it.toLong() }.toLongArray()
        }
        while (tokens.size < maxLength) tokens.add(padToken)

        return tokens.map { it.toLong() }.toLongArray()
    }

    private fun tokenToBpeTokens(token: String): List<Int> {
        val byteSeq = token.toByteArray(Charsets.UTF_8).map { it.toInt() and 0xFF }
        val text = byteSeq.joinToString("") { byteEncoder[it] ?: error("Missing byte encoder entry") }
        val bpeTokens = bpe(text).split(" ")
        return bpeTokens.mapNotNull { vocab[it] }
    }

    private fun bpe(token: String): String {
        var word = token.chunked(1)
        if (word.size == 1) return token

        var pairs = getPairs(word)
        while (true) {
            val bigram = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            val rank = bpeRanks[bigram]
            if (rank == null) break

            val (first, second) = bigram
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                var j = -1
                for (k in i until word.size) {
                    if (word[k] == first) {
                        j = k
                        break
                    }
                }
                if (j == -1 || j == word.size - 1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                newWord.addAll(word.subList(i, j))
                if (word[j] == first && word[j + 1] == second) {
                    newWord.add(first + second)
                    i = j + 2
                } else {
                    newWord.add(word[j])
                    i = j + 1
                }
            }
            word = newWord
            if (word.size == 1) break
            pairs = getPairs(word)
        }

        return word.joinToString(" ")
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        var prev = word.firstOrNull() ?: return pairs
        for (i in 1 until word.size) {
            val cur = word[i]
            pairs.add(prev to cur)
            prev = cur
        }
        return pairs
    }

    private fun loadVocab(context: Context): Map<String, Int> {
        val json = context.assets.open("vocab.json").bufferedReader().use { it.readText() }
        val obj = JSONObject(json)
        val map = mutableMapOf<String, Int>()
        val keys = obj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            map[k] = obj.getInt(k)
        }
        return map
    }

    private fun loadMerges(context: Context): Map<Pair<String, String>, Int> {
        val input = context.assets.open("merges.txt")
        val reader = BufferedReader(InputStreamReader(input))
        val map = mutableMapOf<Pair<String, String>, Int>()
        reader.useLines { lines ->
            lines.drop(1).forEachIndexed { idx, line ->
                val parts = line.split(" ")
                if (parts.size == 2) {
                    map[parts[0] to parts[1]] = idx
                }
            }
        }
        return map
    }

    private fun buildByteEncoder(): Map<Int, String> {
        val bs = mutableListOf<Int>()
        bs.addAll(33..126)
        bs.addAll(161..172)
        bs.addAll(174..255)
        val cs = bs.toMutableList()
        var n = 0
        for (b in 0 until 256) {
            if (!bs.contains(b)) {
                bs.add(b)
                cs.add(256 + n)
                n += 1
            }
        }
        val map = mutableMapOf<Int, String>()
        for (i in bs.indices) {
            map[bs[i]] = String(Character.toChars(cs[i]))
        }
        return map
    }
}
