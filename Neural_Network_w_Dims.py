#Simple neural net backpropigates to find 1s or 0s stored as y.  
#Note the comments above layers. These are matrix dimensions. A 300,1 matrix times a 7,12 matrix = a 7,1 matrix.


import numpy as np 

def sigmoid(x): return 1/(1+np.exp(-x))

def sig_deriv(x): return x*(1-x)

x = np.array([ [0,1,1,0,0],
				[0,1,0,0,0],
				[0,1,1,0,0],
				[1,1,1,0,1],
				[0,1,1,1,0],
				[1,1,0,0,0],
				[0,1,0,0,0] ])

y = np.array([ [1],
				[0],
				[1],
				[0],
				[0],
				[1],
				[0] ])


np.random.seed(22)

weights_0 = (np.random.randn(5, 13))
weights_1 = (np.random.randn(13, 22))
weights_2 = (np.random.randn(22, 33))
weights_3 = (np.random.randn(33, 30))
weights_4 = (np.random.randn(30, 1))

lr = 0.01
for i in range(500000):
	#FORWARD
	#7,13            #7,5 #5,13
	layer_1 = sigmoid(x @ weights_0)
	
	#7,22            #7,13      #13,22
	layer_2 = sigmoid(layer_1 @ weights_1)

	#7,33            #7,22      #22,33
	layer_3 = sigmoid(layer_2 @ weights_2)

	#7,30            #7,33      #33,30
	layer_4 = sigmoid(layer_3 @ weights_3)

	#7,1			 #7,30      #30,1
	layer_5 = sigmoid(layer_4 @ weights_4)


	#CALCULATE ERRORS, DERIVATIVES
	#7,1			    #7,1	  #7,1
	prediction_error = (layer_5 - y)

	#7,1        	   #7,1		        	        #7,1
	prediction_delta = prediction_error * sig_deriv(layer_5)

	#7,12			#7,1		       #1,12
	layer_4_error = prediction_delta @ weights_4.T

	#7,12			#7,12		    		  #7,12
	layer_4_delta = layer_4_error * sig_deriv(layer_4)

	#7,12			#7,1			#1,12
	layer_3_error = layer_4_delta @ weights_3.T

	#7,12			#7,12   				  #7,12
	layer_3_delta = layer_3_error * sig_deriv(layer_3)

	#7,11			#7,12			#12,11
	layer_2_error = layer_3_delta @ weights_2.T

	#7,11			#7,11					  #7,11
	layer_2_delta = layer_2_error * sig_deriv(layer_2)

	#7,10			#7,11			#11,10
	layer_1_error = layer_2_delta @ weights_1.T

	#7,10			#7,10					  #7,10
	layer_1_delta = layer_1_error * sig_deriv(layer_1)


	#UPDATE WEIGHTS
	#300,1			  #7,12		  #7,1
	weights_4 -= lr * layer_4.T @ prediction_delta

	#12,1			  #12,7		  #7,1
	weights_3 -= lr * layer_3.T @ layer_4_delta

	#11,12		      #11,7	      #7,12
	weights_2 -= lr * layer_2.T @ layer_3_delta

	#10,11		      #10,7		  #7,11
	weights_1 -= lr * layer_1.T @ layer_2_delta

	#7,10		      #7,7  #7,10
	weights_0 -= lr * x.T @ layer_1_delta

	if i % 50 == 0:
		print(((prediction_error) ** 2).mean())






