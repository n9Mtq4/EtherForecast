import com.n9mtq4.kotlin.extlib.syntax.def
import java.util.Arrays

/**
 * Created by will on 6/8/17 at 8:30 PM.
 *
 * @author Will "n9Mtq4" Bresnahan
 */
fun main(args: Array<String>) {
	
	val network = NeuralNetwork(3, 2, 3, 1, 5)
	
	for (i in 0..10000) {
		
		val (inputs, correct) = generateXor()
		
		val nInputs = NeuronValues(2)
		nInputs.add(inputs[0].toDouble())
		nInputs.add(inputs[1].toDouble())
		
		val nCorrect = NeuronValues(1)
		nCorrect.add(correct.toDouble())
		network.train(nInputs, nCorrect)
		
	}
	
	test(network)
	
}

fun test(network: NeuralNetwork) {
	
	for (i in 1..20) {
		
		val (inputs, correct) = generateXor()
		
		network.layers.first().apply {
			neurons[0].output = inputs[0].toDouble()
			neurons[1].output = inputs[1].toDouble()
		}
		network.pulse()
		
		val output = network.layers.last().neurons[0].output
		
		println("Input: ${Arrays.toString(inputs)}")
		println("Correct Output: $correct")
		println("Network Output: $output")
		println("Error: ${network.globalError * 100.0}%")
		
	}
	
}

fun Boolean.toInt() = if (this) 1 else 0
fun generateXor() = def {
	val inputs = arrayOf(RANDOM.nextBoolean().toInt(), RANDOM.nextBoolean().toInt())
	val correct = inputs[0] xor inputs[1]
	Pair(inputs, correct)
}

/*
* 
network = Network.new(3, 2, 3, 1, 5)

times_trained = 0
while true
  inputs, correct = generate_xor
  network.train(inputs, correct)
  output = network.layers.last.neurons[0].output.round

  puts "Input: " + inputs.to_s
  puts "Correct Output: " + correct[0].to_s
  puts "Network Output: " + output.to_s
  puts "Error: " + (network.global_error * 100.0).to_s + "%"

  times_trained += 1

  if network.global_error < 0.1
    if test(network)
      break
    end
  end
end

puts "Training complete"
puts "It took " + times_trained.to_s + " pulses"
* */
