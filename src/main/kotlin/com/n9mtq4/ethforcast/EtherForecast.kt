package com.n9mtq4.ethforcast

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions






/**
 * Created by will on 6/8/17 at 9:35 PM.
 *
 * @author Will "n9Mtq4" Bresnahan
 */

const val NUM_INPUTS = 240
const val NUM_OUTPUTS = 7
const val NUM_HIDDEN_ONE = 480
const val NUM_HIDDEN_TWO = 240
const val SEED = 1234
val numRows = 28 // The number of rows of a matrix.
val numColumns = 28 // The number of columns of a matrix.

fun main(args: Array<String>) {
	
//	processData()
	val str = "4.89E-4"
	val d = str.toDouble()
	println(d + 1.0)
	
}

fun runModel() {
	
	val conf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
			.seed(SEED)
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			.iterations(1)
			.learningRate(0.0006)
			.updater(Updater.NESTEROVS).momentum(0.9)
			.regularization(true).l2(1e-4)
			.list()
			.layer(0, DenseLayer.Builder()
					.nIn(NUM_INPUTS) // Number of input datapoints.
					.nOut(NUM_HIDDEN_ONE) // Number of output datapoints.
					.activation(Activation.RELU) // Activation function.
					.weightInit(WeightInit.XAVIER) // Weight initialization.
					.build())
			.layer(0, DenseLayer.Builder()
					.nIn(NUM_INPUTS) // Number of input datapoints.
					.nOut(NUM_HIDDEN_ONE) // Number of output datapoints.
					.activation(Activation.RELU) // Activation function.
					.weightInit(WeightInit.XAVIER) // Weight initialization.
					.build())
			.layer(1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
					.nIn(NUM_HIDDEN_TWO)
					.nOut(NUM_HIDDEN_ONE)
					.activation(Activation.SOFTMAX)
					.weightInit(WeightInit.XAVIER)
					.build())
			.pretrain(false).backprop(true)
			.build()
	
	val model = MultiLayerNetwork(conf)
	model.init()
	//print the score with every 1 iteration
	model.setListeners(ScoreIterationListener(1))
	
/*	println("Train model....")
	for (i in 0..15 - 1) {
		model.fit(mnistTrain)
	}


//	model.output() // how to run the network
	
	println("Evaluate model....")
	val eval = Evaluation(10) //create an evaluation object with 10 possible classes
	while (mnistTest.hasNext()) {
		val next = mnistTest.next()
		val output = model.output(next.featureMatrix) //get the networks prediction
		eval.eval(next.labels, output) //check the prediction against the true class
	}
	
	println(eval.stats())
	println("****************Example finished********************")*/
	
}
