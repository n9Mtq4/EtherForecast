package com.n9mtq4.ethforcast

import com.n9mtq4.kotlin.extlib.io.open

/**
 * Created by will on 6/9/17 at 10:45 PM.
 *
 * @author Will "n9Mtq4" Bresnahan
 */

typealias EthData = Array<EthPricePoint>
data class EthPricePoint(val date: Int, val deltaPercent: Double, val volume: Double)

fun loadData(filePath: String): EthData = open(filePath, "r").use { file -> 
	file.readLines()?.map { line ->
		val tokens = line.split(",").map { it.trim() }
		val date = tokens[0].toInt()
		val deltaPercent = tokens[1].toDouble()
		val volume = tokens[2].toDouble()
		EthPricePoint(date, deltaPercent, volume)
	}?.toTypedArray() ?: EthData(0) { EthPricePoint(-1, -1.0, -1.0) }
}

