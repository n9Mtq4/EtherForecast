package com.n9mtq4.ethforcast

import com.n9mtq4.kotlin.extlib.pst
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File


/**
 * Created by will on 6/8/17 at 9:35 PM.
 *
 * @author Will "n9Mtq4" Bresnahan
 */

const val DAYS_INPUTS = 30
const val DAYS_OUTPUTS = 7
const val POINTS_PER_DAY = 6
const val POINTS_SIZE = 2

const val NUM_INPUTS = DAYS_INPUTS * POINTS_PER_DAY * POINTS_SIZE
const val NUM_HIDDEN_ONE = NUM_INPUTS * POINTS_SIZE
const val NUM_HIDDEN_TWO = NUM_INPUTS
const val NUM_OUTPUTS = DAYS_OUTPUTS * POINTS_PER_DAY

const val SEED = 123

fun main(args: Array<String>) {
	
//	processData()
//	xorTest()
	runModel()
	
}

fun xorTest() {
	
	// list off input values, 4 training samples with data for 2
	// input-neurons each
	val input = Nd4j.zeros(4, 2)
	
	// correspondending list with expected output values, 4 training samples
	// with data for 2 output-neurons each
	val labels = Nd4j.zeros(4, 2)
	
	// create first dataset
	// when first input=0 and second input=0
	input.putScalar(intArrayOf(0, 0), 0)
	input.putScalar(intArrayOf(0, 1), 0)
	// then the first output fires for false, and the second is 0 (see class
	// comment)
	labels.putScalar(intArrayOf(0, 0), 1)
	labels.putScalar(intArrayOf(0, 1), 0)
	
	// when first input=1 and second input=0
	input.putScalar(intArrayOf(1, 0), 1)
	input.putScalar(intArrayOf(1, 1), 0)
	// then xor is true, therefore the second output neuron fires
	labels.putScalar(intArrayOf(1, 0), 0)
	labels.putScalar(intArrayOf(1, 1), 1)
	
	// same as above
	input.putScalar(intArrayOf(2, 0), 0)
	input.putScalar(intArrayOf(2, 1), 1)
	labels.putScalar(intArrayOf(2, 0), 0)
	labels.putScalar(intArrayOf(2, 1), 1)
	
	// when both inputs fire, xor is false again - the first output should
	// fire
	input.putScalar(intArrayOf(3, 0), 1)
	input.putScalar(intArrayOf(3, 1), 1)
	labels.putScalar(intArrayOf(3, 0), 1)
	labels.putScalar(intArrayOf(3, 1), 0)
	
	// create dataset object
	val ds = DataSet(input, labels)
	
	// Set up network configuration
	val builder = NeuralNetConfiguration.Builder()
	// how often should the training set be run, we need something above
	// 1000, or a higher learning-rate - found this values just by trial and
	// error
	builder.iterations(10000)
	// learning rate
	builder.learningRate(0.1)
	// fixed seed for the random generator, so any run of this program
	// brings the same results - may not work if you do something like
	// ds.shuffle()
	builder.seed(123)
	// not applicable, this network is to small - but for bigger networks it
	// can help that the network will not only recite the training data
	builder.useDropConnect(false)
	// a standard algorithm for moving on the error-plane, this one works
	// best for me, LINE_GRADIENT_DESCENT or CONJUGATE_GRADIENT can do the
	// job, too - it's an empirical value which one matches best to
	// your problem
	builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	// init the bias with 0 - empirical value, too
	builder.biasInit(0.0)
	// from "http://deeplearning4j.org/architecture": The networks can
	// process the input more quickly and more accurately by ingesting
	// minibatches 5-10 elements at a time in parallel.
	// this example runs better without, because the dataset is smaller than
	// the mini batch size
	builder.miniBatch(false)
	
	// create a multilayer network with 2 layers (including the output
	// layer, excluding the input payer)
	val listBuilder = builder.list()
	
	val hiddenLayerBuilder = DenseLayer.Builder()
	// two input connections - simultaneously defines the number of input
	// neurons, because it's the first non-input-layer
	hiddenLayerBuilder.nIn(2)
	// number of outgooing connections, nOut simultaneously defines the
	// number of neurons in this layer
	hiddenLayerBuilder.nOut(4)
	// put the output through the sigmoid function, to cap the output
	// valuebetween 0 and 1
	hiddenLayerBuilder.activation(Activation.SIGMOID)
	// random initialize weights with values between 0 and 1
	hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION)
	hiddenLayerBuilder.dist(UniformDistribution(0.0, 1.0))
	
	// build and set as layer 0
	listBuilder.layer(0, hiddenLayerBuilder.build())
	
	// MCXENT or NEGATIVELOGLIKELIHOOD (both are mathematically equivalent) work ok for this example - this
	// function calculates the error-value (aka 'cost' or 'loss function value'), and quantifies the goodness
	// or badness of a prediction, in a differentiable way
	// For classification (with mutually exclusive classes, like here), use multiclass cross entropy, in conjunction
	// with softmax activation function
	val outputLayerBuilder = OutputLayer.Builder(LossFunctions.LossFunction.MSE)
	// must be the same amout as neurons in the layer before
	outputLayerBuilder.nIn(4)
	// two neurons in this layer
	outputLayerBuilder.nOut(2)
	outputLayerBuilder.activation(Activation.SOFTMAX)
	outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION)
	outputLayerBuilder.dist(UniformDistribution(0.0, 1.0))
	listBuilder.layer(1, outputLayerBuilder.build())
	
	// no pretrain phase for this network
	listBuilder.pretrain(false)
	
	// seems to be mandatory
	// according to agibsonccc: You typically only use that with
	// pretrain(true) when you want to do pretrain/finetune without changing
	// the previous layers finetuned weights that's for autoencoders and
	// rbms
	listBuilder.backprop(true)
	
	// build and init the network, will check if everything is configured
	// correct
	val conf = listBuilder.build()
	val net = MultiLayerNetwork(conf)
	net.init()
	
	// add an listener which outputs the error every 100 parameter updates
	net.setListeners(ScoreIterationListener(100))
	
	// C&P from GravesLSTMCharModellingExample
	// Print the number of parameters in the network (and for each layer)
	val layers = net.layers
	var totalNumParams = 0
	for (i in layers.indices) {
		val nParams = layers[i].numParams()
		println("Number of parameters in layer $i: $nParams")
		totalNumParams += nParams
	}
	println("Total number of network parameters: " + totalNumParams)
	
	// here the actual learning takes place
	net.fit(ds)
	
	// create output for every training sample
	val output = net.output(ds.getFeatureMatrix())
	println(output)
	
	// let Evaluation prints stats how often the right output had the
	// highest value
	val eval = Evaluation(2)
	eval.eval(ds.getLabels(), output)
	System.out.println(eval.stats())
	
	println(net.output(Nd4j.create(arrayOf(1.0, 0.0).toDoubleArray())))
	println(net.output(Nd4j.create(arrayOf(1.0, 1.0).toDoubleArray())))
	println(net.output(Nd4j.create(arrayOf(0.0, 0.0).toDoubleArray())))
	println(net.output(Nd4j.create(arrayOf(0.0, 1.0).toDoubleArray())))
	
	
}

fun runModel() {
	
	val conf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
			.seed(SEED)
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			.iterations(1)
			.learningRate(0.0006)
			.updater(Updater.NESTEROVS).momentum(0.9)
			.useDropConnect(false) // set to true for better results?
			.miniBatch(true) // set to true / false for better performance
			.regularization(true).l2(1e-4)
			.list()
			.layer(0, DenseLayer.Builder()
					.nIn(NUM_INPUTS) // Number of input datapoints.
					.nOut(NUM_HIDDEN_ONE) // Number of output datapoints.
					.activation(Activation.SIGMOID) // Activation function.
					.weightInit(WeightInit.XAVIER) // Weight initialization.
					.build())
			.layer(0, DenseLayer.Builder()
					.nIn(NUM_HIDDEN_ONE) // Number of input datapoints.
					.nOut(NUM_HIDDEN_TWO) // Number of output datapoints.
					.activation(Activation.SIGMOID) // Activation function. // default: relu
					.weightInit(WeightInit.XAVIER) // Weight initialization.
					.build())
			.layer(1, OutputLayer.Builder(LossFunctions.LossFunction.MSE)
					.nIn(NUM_HIDDEN_TWO)
					.nOut(NUM_OUTPUTS)
					.activation(Activation.SOFTMAX)
					.weightInit(WeightInit.XAVIER)
					.build())
			.pretrain(false).backprop(true)
			.build()
	
	val model = MultiLayerNetwork(conf)
	model.init()
	//print the score with every 1 iteration
//	model.setListeners(ScoreIterationListener(1))
	
	val ethData = loadData("data/eth_training_average.csv")
	val slope = (ethData[1].date - ethData[0].date)
	val yint = ethData.first().date
	val stepVal = DAYS_INPUTS * POINTS_PER_DAY
	
	for (i in 0..ethData.size step stepVal) {
		pst {
			
			println("Next iteration")
			
			for (d in 0..stepVal) {
				
				val point = ethData[i + d]
				if (point.date != yint + (i + d) * slope) println("Warning invalid date!")
				
				val inputData = getInputData(point.date, ethData)
				val outputData = getOutputData(point.date, ethData)
				
				model.fit(inputData, outputData)
				
			}
			
		}
	}
	
	
	save(model, "data/model_average.zip")
	
}

fun getInputData(endTime: Int, data: EthData): INDArray {
	
	val dArray = DoubleArray(NUM_INPUTS) { 0.0 }
	val endIndex = data.indexOfFirst { it.date == endTime }
	for (i in endIndex..endIndex + NUM_INPUTS / POINTS_SIZE - 1) {
		val etherDataPoint = data[i]
		dArray[POINTS_SIZE * i] = percentAdjustment(etherDataPoint.deltaPercent)
		dArray[POINTS_SIZE * i + 1] = volumeAdjustment(etherDataPoint.volume)
	}
	
	return Nd4j.create(dArray)
	
}

fun getOutputData(inputEndTime: Int, data: EthData): INDArray {
	
	val dArray = DoubleArray(NUM_OUTPUTS) { 0.0 }
	val endIndex = data.indexOfFirst { it.date == inputEndTime }
	((endIndex + 1)..(endIndex + NUM_OUTPUTS) - 1).forEach { 
		dArray[it] = percentAdjustment(data[it].deltaPercent)
	}
	
	return Nd4j.create(dArray)
	
}

fun percentAdjustment(percent: Double) = percent / 10.0
fun volumeAdjustment(volume: Double) = volume / 1.0e7
fun percentAdjustmentInverse(adjPercent: Double) = adjPercent * 10.0

fun save(net: MultiLayerNetwork, filePath: String) {
	val locationToSave = File(filePath) //Where to save the network. Note: the file is in .zip format - can be opened externally
	val saveUpdater = true //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
	ModelSerializer.writeModel(net, locationToSave, saveUpdater)
	
}

fun load(filePath: String): MultiLayerNetwork {
	val restored = ModelSerializer.restoreMultiLayerNetwork(File(filePath))
	return restored
}
