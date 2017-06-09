
import com.n9mtq4.kotlin.extlib.math.pow
import com.n9mtq4.kotlin.extlib.math.sqrt
import java.util.Random

/**
 * Created by will on 6/8/17 at 5:25 PM.
 *
 * @author Will "n9Mtq4" Bresnahan
 */

const val LEARNING_RATE = 0.7
const val MOMENTUM = 0.3

val RANDOM = Random()

typealias ConnectionList = ArrayList<Connection>
typealias NeuronList = ArrayList<Neuron>
typealias LayerList = ArrayList<Layer>
typealias NeuronValues = ArrayList<Double>

/**
 * Sigmoid function
 * */
fun activation(input: Double): Double = 1 / (1 + Math.exp(-input))

/**
 * Derivative of sigmoid function
 * */
fun dActivation(input: Double): Double {
	val s = activation(input)
	return s * (1.0 - s)
}

/**
 * Generate a random double within range
 * */
fun rand(min: Double, max: Double): Double = min + (max - min) * RANDOM.nextDouble()

class NeuralNetwork(val layerCount: Int, val inputNeuronCount: Int, val hiddenNeuronCount: Int, val outputNeuronCount: Int, val updateAfter: Int) {
	
	var globalError: Double
	var onUpdate: Int
	
	var beta: Double
	
	var layers: LayerList = LayerList(3)
	var connections: ConnectionList = ConnectionList()
	var neurons: NeuronList = NeuronList()
	
	/*
	*     @global_error = 1 GOOD
    @update_after = update_after GOOD
    @on_update = 0 GOOD
    generate_layers(layer_count, input_neuron_count, hidden_neuron_count, output_neuron_count)
    generate_connections

    get_all_neurons
    get_all_connections

    calculate_beta(input_neuron_count, hidden_neuron_count)
    initial_weight_adjust
	* */
	
	init {
		
		this.globalError = 1.0
		this.onUpdate = 0
		this.beta = 0.0 // not in jakes code
		
		generateLayers(layerCount, inputNeuronCount, hiddenNeuronCount, outputNeuronCount)
		generateConnections()
		
		getAllNeurons()
		getAllConnections()
		
		calculateBeta(inputNeuronCount, hiddenNeuronCount)
		initialWeightAdjust()
		
	}
	
	fun pulse() {
		layers.forEach { it.pulse() }
	}
	
	fun train(inputs: NeuronValues, desiredResults: NeuronValues) {
		
		layers.first().neurons.forEachIndexed { index, neuron -> 
			neuron.output = inputs[index]
		}
		
		pulse()
		calculateGlobalError(desiredResults)
		
		this.onUpdate++
		backPropagation(desiredResults)
		
	}
	
	private fun backPropagation(desiredResults: NeuronValues) {
		
		calculateDeltas(desiredResults)
		
		connections.forEach { connection ->
			
			connection.calculateGradient()
			
			if (onUpdate >= updateAfter) {
				connection.calculateDeltaChange()
				connection.updateWeight()
			}
			
		}
		
		if (onUpdate >= updateAfter) onUpdate = 0 // not in jakes code | has it in the if inside the for
		
	}
	
	/*
	*   def calculate_beta (input_neuron_count, hidden_neuron_count)
    @beta = 0.7 * (hidden_neuron_count ** (1.0 / input_neuron_count))
  end

	* */
	private fun calculateBeta(inputNeuronCount: Int, hiddenNeuronCount: Int) {
		this.beta = 0.7 * (hiddenNeuronCount.pow(1.0 / inputNeuronCount))
	}
	
	private fun calculateDeltas(desiredResults: NeuronValues) {
		
		// calculate deltas for output layer
		layers.last().neurons.forEachIndexed { index, neuron -> 
			val error = neuron.output - desiredResults[index]
			neuron.delta = -error * dActivation(neuron.sum)
		}
		
		// calculate deltas for other layers
		neurons.forEach { it.calculateDelta() }
		
	}
	
	private fun calculateGlobalError(desiredResults: NeuronValues) {
		
		val sum = layers.last().neurons.mapIndexed { index, neuron -> (desiredResults[index] - neuron.output).pow(2) }.sum()
		
		this.globalError = sum / desiredResults.size
		
	}
	
	private fun getAllNeurons() {
		
		neurons.clear()
		
		layers.flatMapTo(neurons) { it.neurons } // TODO: may be problematic
//		layers.forEach { neurons.addAll(it.neurons) }
		
	}
	
	private fun getAllConnections() {
		
		// TODO: lots of changes from jake's code. could cause many errors
		
		connections.clear()
		
		// doesn't include bias layer stuff
		// layers.filterNot { it.isOutputLayer }.flatMapTo(connections) { it.neurons.flatMap { it.outputConnections } }
		
		layers.forEach { layer ->
			if (!layer.isOutputLayer) {
				layer.neurons.flatMapTo(connections) { it.outputConnections }
			}
			if (!layer.isInputLayer) {
				val bias = layer.bias
				bias.outputConnections.mapTo(connections) { it }
			}
		}
		
	}
	
	private fun generateLayers(layerCount: Int, inputNeuronCount: Int, hiddenNeuronCount: Int, outputNeuronCount: Int) {
		
//		this.layers = LayerList(layerCount)
/*		this.layers.clear()
		this.layers[0] = Layer(inputNeuronCount, isInputLayer = true)
		this.layers[layerCount - 1] = Layer(outputNeuronCount, isOutputLayer = true)
		
		// skip input and output layers
		for (i in 1..layerCount - 2) {
			this.layers[i] = Layer(hiddenNeuronCount)
		}*/
		
		this.layers.add(Layer(inputNeuronCount, isInputLayer = true))
		for (i in 1..layerCount - 2) {
			this.layers.add(Layer(hiddenNeuronCount))
		}
		this.layers.add(Layer(outputNeuronCount, isOutputLayer = true))
		
	}
	
	private fun generateConnections() {
		layers.forEachIndexed { index, layer -> 
			if (index == layers.size - 1) return@forEachIndexed // not in jakes code
			layer.generateConnections(layers[index + 1])
		}
	}
	
	private fun initialWeightAdjust() {
		
		// only do first hidden layer
		this.layers[1].neurons.forEach { neuron ->
			
			neuron.calculateNorm()
			
			neuron.inputConnections.forEach { it.initWeightAdjust(this.beta) }
			
		}
		
		this.layers[1].bias.apply { // bias neurons for n layer are a part of n layer
			calculateNorm()
			outputConnections.forEach { it.initWeightAdjust(this@NeuralNetwork.beta) }
		}
		
	}
	
}

class Neuron(var layer: Layer) {
	
	var inputConnections: ConnectionList
	var outputConnections: ConnectionList
	var delta: Double
	var output: Double
	var sum: Double
	var norm: Double
	
	init {
		
		inputConnections = ConnectionList()
		outputConnections = ConnectionList()
		delta = 0.0
		output = 1.0
		sum = 0.0
		
		norm = 0.0 // TODO: not in jakes code
		
	}
	
	fun calculateDelta() {
		if (!layer.isOutputLayer && !layer.isInputLayer) {
			// skip output (already done) and input (no need). This also skips biases
			this.delta = 0.0
			/*outputConnections.forEach { connection ->
				val neuron = connection.outputNeuron
				this.delta += neuron.delta * connection.weight
			}*/
			
			delta = outputConnections.map { it.outputNeuron.delta * it.weight }.sum()
			
		}
		
		this.delta *= dActivation(this.sum)
		
	}
	
	fun calculateNorm() {
/*		this.norm = 0.0
		inputConnections.forEach { connection ->
			this.norm += connection.weight.pow(2)
		}
		this.norm = norm.sqrt()*/
		
		this.norm = inputConnections.map { it.weight.pow(2) }.sum().sqrt()
	}
	
	fun pulse() {
		
		// clear
		this.output = 0.0
		this.sum = 0.0
		
		// set sum
		this.sum = inputConnections.map { it.calculateValue() }.sum()
		
		// get bias connection
		sum += layer.bias.outputConnections.filter { it.outputNeuron == this }.map { it.calculateValue() }.first()
		
		// activation function on output
		this.output = activation(sum)
		
	}
	
	fun generateOutputConnections(nextLayer: Layer) {
		
		nextLayer.neurons.forEach { neuron ->
			val connection = Connection(this, neuron)
			neuron.inputConnections.add(connection)
			this.outputConnections.add(connection)
		}
		
	}
	
}

class Layer(val neuronCount: Int, val isInputLayer: Boolean = false, val isOutputLayer: Boolean = false) {
	
	val neurons: NeuronList = NeuronList(neuronCount)
	lateinit var bias: Neuron
	
	init {
		generateNeurons()
		if (!isInputLayer) {
			generateBias()
		}
	}
	
	fun pulse() {
		if (!isInputLayer) {
			neurons.forEach { it.pulse() }
		}
	}
	
	fun generateConnections(nextLayer: Layer) {
		if (!isOutputLayer) { // don't create output connections if there is no neuron to connect to
			neurons.forEach { it.generateOutputConnections(nextLayer) }
		}
	}
	
	private fun generateNeurons() {
		
		neurons.clear()
		
		for (i in 0..neuronCount - 1) {
			this.neurons.add(i, Neuron(this))
		}
		
	}
	
	private fun generateBias() {
		this.bias = Neuron(this)
		bias.output = 1.0
		bias.generateOutputConnections(this)
	}
	
}

class Connection(var inputNeuron: Neuron, var outputNeuron: Neuron) {
	
	var weight: Double
	var gradient: Double
	var deltaChange: Double
	
	init {
		
		weight = rand(-1.0, 1.0)
		deltaChange = 0.0
		gradient = 0.0
		
	}
	
	fun calculateValue() = inputNeuron.output * weight
	
	fun initWeightAdjust(beta: Double) {
		weight = (beta * weight) / outputNeuron.norm
	}
	
	fun calculateGradient() {
		gradient += inputNeuron.output * outputNeuron.delta
	}
	
	fun calculateDeltaChange() {
		deltaChange = (LEARNING_RATE * gradient) + (MOMENTUM * deltaChange)
		gradient = 0.0
	}
	
	fun updateWeight() {
		weight += deltaChange
	}
	
}
