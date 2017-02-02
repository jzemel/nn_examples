#simple conceptual Neural Net (this is actually Siraj Naval's)
from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		#seed the generator so it generates the same numbers every time
		random.seed(1)

		#model a single neuron with 3 input and 1 output connection
		#assign random weights to a 3x1 matrix with vals from -1 to 1 (and mean 0)
		self.synaptic_weights = 2 * random.random((3,1)) - 1

	#sigmoid defines an s shaped curve that normalizes inputs between 0,1
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1-x)

	def train(self, training_set_inputs, training_set_outputs, num_iterations):
		for iteration in range(0,num_iterations):
			#pass training set through network
			output = self.predict(training_set_inputs)
			
			#calculate error
			error = training_set_outputs - output

			#multiply error by input and by gradient of sigmoid curve
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

			#adjust the weights
			self.synaptic_weights += adjustment


	def predict(self, inputs):
		#pass inputs through our single neuron network
		return self.__sigmoid(dot(inputs, self.synaptic_weights))


def run():
	
	#initialize single neuron network
	neural_network = NeuralNetwork()

	print('Random starting synaptic weights: ')
	print(neural_network.synaptic_weights)

	#The training set: 4 examples of 3 input vals and 1 output val
	training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T

	#train the neural net 10,000 times
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print('New synaptic weights after training: ')
	print(neural_network.synaptic_weights)

	print('predicting [1,0,0]: ')
	print(neural_network.predict([[1,0,0]]))


if __name__ == '__main__':
	run()