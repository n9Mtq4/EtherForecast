package com.n9mtq4.ethforcast

import com.google.gson.JsonParser
import com.n9mtq4.kotlin.extlib.ignore
import com.n9mtq4.kotlin.extlib.io.open


/**
 * Created by will on 6/9/17 at 11:25 AM.
 *
 * @author Will "n9Mtq4" Bresnahan
 */


fun processData() {
	
	genAverageData()
	genPessimisticData()
	genOptimisticData()
	
}

fun percent(first: Double, second: Double) = 1 - (first / second)

fun genAverageData() {
	
	val file = open("data/eth_training_raw.json", "r")
	
	val outFile = open("data/eth_training_average.csv", "w")
	
	val jsonString = file.read() ?: ""
	val jelement = JsonParser().parse(jsonString)
	val jArray = jelement.asJsonArray
	
	jArray.forEachIndexed { index, jsonElement ->
		ignore {
			val jsObj = jsonElement.asJsonObject
			val date = jsObj.get("date")
			val meanBefore = jArray[index - 1].asJsonObject.get("weightedAverage").asDouble
			val meanAfter = jsObj.get("weightedAverage").asDouble
//			val diff = maxAfter - minBefore
//			val percent = meanAfter / meanBefore
			val percent = percent(meanAfter, meanBefore)
			val volume = jsObj.get("volume").asDouble
			outFile.writeln("$date, $percent, $volume")
		}
	}
	
	outFile.flushWriter()
	outFile.close()
	
	file.close()
	
	println("Done with average data")
	
}

fun genPessimisticData() {
	
	val file = open("data/eth_training_raw.json", "r")
	
	val outFile = open("data/eth_training_pessimistic.csv", "w")
	
	val jsonString = file.read() ?: ""
	val jelement = JsonParser().parse(jsonString)
	val jArray = jelement.asJsonArray
	
	jArray.forEachIndexed { index, jsonElement ->
		ignore {
			val jsObj = jsonElement.asJsonObject
			val date = jsObj.get("date")
			val maxBefore = jArray[index - 1].asJsonObject.get("high").asDouble
			val minAfter = jsObj.get("low").asDouble
//			val diff = maxAfter - minBefore
//			val percent = minAfter / maxBefore
			val percent = percent(minAfter, maxBefore)
			val volume = jsObj.get("volume").asDouble
			outFile.writeln("$date, $percent, $volume")
		}
	}
	
	outFile.flushWriter()
	outFile.close()
	
	file.close()
	
	println("Done with pessimistic data")
	
}

fun genOptimisticData() {
	
	val file = open("data/eth_training_raw.json", "r")
	
	val outFile = open("data/eth_training_optimistic.csv", "w")
	
	val jsonString = file.read() ?: ""
	val jelement = JsonParser().parse(jsonString)
	val jArray = jelement.asJsonArray
	
	jArray.forEachIndexed { index, jsonElement ->
		ignore {
			val jsObj = jsonElement.asJsonObject
			val date = jsObj.get("date")
			val minBefore = jArray[index - 1].asJsonObject.get("low").asDouble
			val maxAfter = jsObj.get("high").asDouble
//			val diff = maxAfter - minBefore
//			val percent = maxAfter / minBefore
			val percent = percent(maxAfter, minBefore)
			val volume = jsObj.get("volume").asDouble
			outFile.writeln("$date, $percent, $volume")
		}
	}
	
	outFile.flushWriter()
	outFile.close()
	
	file.close()
	
	println("Done with optimistic data")
	
}
