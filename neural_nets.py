
import numpy as np
import h5py as h5
import math
import matplotlib.pyplot as plt

def load_data(fname,dataset):
    """
    loads a specified data set from a hdf5 file

    :param fname: location of the hdf5 file from where data is to be loaded ex: folder/fname.hdf5
    :param dataset: name of the dataset to be loaded as it appears in the hdf5 file
    :return: returns the specified dataset in numpy array form
    """
    def get_obj(name,obj):    # this function will be used by the f.visititems which will iterate through
        if dataset in name:   # the contents of the file and when it will encounter an item with the same name as or
            return obj        # consisting of 'dataset', it will return the corresponding item

    with h5.File(fname,'r') as f: # open the file with read permission
        return np.array(f.visititems(get_obj))

def standardize_image(x):
    """
    Converts the numpy array x of the form (m,w,h,3)(where m is number of images, w,h are width and height of each
    image and 3 corresponds to image data) into a numpy array of the form (w*h*3,m) to be used as input for neural nets
    :param x: corresponds to image data as mentioned above
    :return: returns  a numpy array to be processed by neural nets as mentioned above
    """
    std_x=x/255
    std_x=std_x.reshape((x.shape[0],-1)).T

    return std_x

def dictionary_to_vector(parameters):
    """
    Function to convert input dictionary parameters (keys:- 'W1','b1','W2','b2',....) into a one column vector
    :param parameters: parameters is a dict input as mentioned above
    :return: theta, a one column vector
    """
    count =0
    for param in parameters:
        new_vector = np.reshape(parameters[param],(-1,1))

        if count==0:
            theta = new_vector
        else:
            theta = np.concatenate((theta,new_vector), axis=0)

        count= count + 1

    return theta

def vector_to_dictionary(theta, layer_dims):
    """
    Converts a one column vector into a dictionary of parameters for a neural net
    :param theta:   The one column vector to be converted
    :param layer_dims: A list containing the number of nodes in each layer of the neural net
    :return:  parameters, a dictionary of parameters W and b for each layer of the neural net
    """
    parameters = {}
    L=len(layer_dims)
    index = 0

    for l in range(1,L):  # iterating over all the layers, l is the current layer number
        nW = layer_dims[l] * layer_dims[l - 1]  # no. of elements in matrix Wl
        nb = layer_dims[l]                      # no. of elements in matrix bl
        parameters['W'+str(l)]=theta[index: index + nW].reshape((layer_dims[l],layer_dims[l-1]))
        parameters['b'+str(l)]=theta[index + nW: index + nW + nb].reshape((layer_dims[l],1))
        index = index + nW + nb

    return parameters


class Logistic_Regression():
    def __init__(self,X,Y,X_test,Y_test,learning_rate,iterations):
        """
        Takes the following input:

        1.)X-----The training data set of the shape(w*h*3,m). It should consist of image data in the np.array format.
                 m is the number of training examples,w,h are the width and height of the image, the quantity 3 stands for rgb data

        2.)Y-----This is the result array that contains the results for the training data set in terms of values 1 or 0. The model
                 will learn from these values
                 It should be of the shape(1,m) where m is the number of training examples

        3.)X_test, Y_test ----- testing set data of the same format as X and Y stated previously

        4.)learning_rate------- rate at which the model is going to learn i.e update its W and b parameters

        5.)iterations----------- number of iterations for which the model is expected to learn

        """
        self.X_test=X_test  
        self.Y_test=Y_test
        self.X=X
        self.m=X.shape[1]   #no. of training examples
        #self.n=X.shape[1]*X.shape[2]*3  #no. of features/rgb pixels stacked up per training example
        #self.X=X.reshape(self.m,-1).T   # reshaping the training data so that each column represents one training example with its rgb pixels stacked up
        #self.X=self.X/255   #standardizing the data for images
        self.Y=Y
        self.W=np.zeros((self.X.shape[0],1))  #Initializing parameters W and b
        self.b=0
        self.rate=learning_rate
        self.itr=iterations

    def sigmoid(self,z):           # will be used for computing the Activation given by A=sigmoid(z)
        """
        will be used for computing the Activation given by A=sigmoid(z)
        returns the value sigmoid(z) for a given input z(numpy array)
        """
        result = 1/(1+np.exp(-z))
        return result

    def forward_prop(self,X_data):  # Computes the forward propagation on all training examples
        """
        Computes the forward propagation for a given dataset
        Takes the inputs:
        X_data---- dataset on which forward prop is to be performed ,
                   should be of the form(w*h*3,m), where m is number of images or examples, w,h are the width and height of the image, 3 stands for rgb
        """
        W=self.W
        X=X_data
        b=self.b
        self.Z=np.dot(W.T,X)+b               # computes Z=np.dot(W.T,X)+b
        self.A=self.sigmoid(self.Z)          # computes A=sigmoid(Z)

    def compute_cost(self):     #computes the cost for all training examples
        """
        computes the cost for all training examples
        """
        self.Cost= -(np.dot(self.Y,np.log(self.A).T)+np.dot(1-self.Y,np.log(1-self.A).T))/self.m
        self.Cost=np.squeeze(self.Cost)   #for calculation purposes so that the new shape is of the form ()

    def backward_prop(self):         #computes backward propagation for all training examples/calculates values of dW and db
        """
        computes backward propagation for all training examples/calculates values of dW and db to be used in gradient descent
        """
        self.dW=np.dot(self.X,(self.A-self.Y).T)/self.m                            
        self.db=np.sum(self.A-self.Y,axis=1,keepdims=True)/self.m

    def update_params(self):       # computes gradient descent
        """
        Carries out gradient descent i.e updates parameters W and b in order to minimize cost
        """
        self.W=self.W-(self.rate*self.dW)
        self.b=self.b-(self.rate*self.db)
        
    def learn(self,print_cost=False,per_itr=100):   #computes forward prop, then backward prop and then updates the parameters to minimize cost
        """
        This is how the model learns.It first computes forward prop, calculates the cost function for all training examples, then computes
        backward prop to find out the values of dW and db and then these values are used for updating the parameters W and b (gradient descent).
        Also prints the value of cost after every 100 iterations, and the training set and testing set accuracy if print_cost is set to True in the argument
        :param per_itr : after how many iterations cost is to be printed
        """
        
        self.costs=[]                   # will store all the values of costs after every 100 iterations
        for i in range(self.itr):
            self.forward_prop(self.X)
            self.compute_cost()
            self.backward_prop()
            self.update_params()

            if i%per_itr==0:
                self.costs.append(self.Cost)   

            if print_cost and i%per_itr==0:
                print("Cost after iteration {} : {}".format(i,self.Cost))  #prints cost after every 100 iterartions

        self.accuracy_train=self.accuracy(self.X,self.Y)
        self.accuracy_test=self.accuracy(self.X_test,self.Y_test)
        print("\n----- Training set Accuracy : {}% --------".format(self.accuracy_train))
        print("----- Testing set Accuracy : {}% --------".format(self.accuracy_test))


    def predict(self,X_test_data):       #predicts the outputs for a given set of data
        """
        predicts and returns the outputs for a given set of data X_test
        X_test should be of the form(w*h*m,3), where m is number of images or examples, w,h are the width and height of the image, 3 stands for rgb
        The returned output is of the form (1,m) containing predictions for each example
        """
        X_test=X_test_data
        m=X_test.shape[1]
        #X_test=X_test.reshape(m,-1).T
        #X_test=X_test/255
        Y_prediction=np.zeros((1,m))
        self.forward_prop(X_test)

        Y_prediction=np.where(self.A>0.5,1,0)

        return Y_prediction

    def accuracy(self,X_test,Y_test):      #returns the accuracy of the model for a given testing data set(X_test,Y_test), both should be provided
        """
        returns the accuracy of the model for a given testing data set(X_test,Y_test), both should be provided
        X_test should be of the form(w*h*3,m), where m is number of images or examples, w,h are the width and height of the image, 3 stands for rgb
        Y_test should be of the form(1,m)- contains values 1 or 0 for each example provided in X_test
        """
        predictions=self.predict(X_test)
        temp=np.abs(Y_test-predictions)
        temp=float(np.squeeze(np.sum(temp)))
        m=X_test.shape[1]
        accuracy=100-((temp/m)*100)
        return accuracy

    def plot_learning_curve(self):
        """
        Plots the learning curve for the model with x axis as every 100 iterations and y axis as the corresponding value of the cost at the specified iteration
        Use the command ----plt.show()---- to be able to see the curve after calling this function
        """
        costs=np.squeeze(self.costs)
        plt.plot(costs,label='{}({},{})'.format(self.rate,self.accuracy_train,self.accuracy_test))
        plt.ylabel('Cost')
        plt.xlabel('Iterations(per hundreds)')
        plt.legend()
        plt.title('----Simple Logistic Regression----\n----Legend----[ Rate(train accuracy,test accuracy) ] ')

class shallow_net():    # Neural net with one hidden layer, and one node in the output layer
    def __init__(self,x_train,y_train,x_test,y_test,n_h,learning_rate,iterations):
        """
        :param x_train: the training dataset on which the model will learn
                        should be of the form,(n_x,m) where n_x is the number of features in each training example
                        and m is the total number of training examples
        :param y_train: the output for the training data set. Should be of the shape (1,m) or (n_y,1,m) with values 1 or 0
                        ,where n_y will be the number of nodes in output layer
        :param x_test:  the testing data set, should be of the form (n_x,m)
        :param y_test:  the output for the testing data set. Should be of the shape (1,m) or (n_y,1,m) with values 1 or 0
        :param n_h:     the number of nodes in the hidden layer
        :param learning_rate: the rate at which params will be updated(gradient descent)
        :param iterations: total number of iterations over the training set for which the model will be learning
        """
        self.x=x_train
        self.y=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.m=x_train.shape[1]
        self.n_x=x_train.shape[0]
        self.n_h=n_h
        self.n_y=y_train.shape[0]
        self.W1=np.random.randn(self.n_h,self.n_x)*0.01   # Initializing the parameters for hidden and output layers
        self.b1=np.zeros((self.n_h,1))
        self.W2=np.random.randn(self.n_y,self.n_h)*0.01
        self.b2=np.zeros((self.n_y,1))
        self.rate=learning_rate
        self.itr=iterations
        self.costs=[]  # will be used for storing value of cost after every 1000 iterations

    def sigmoid(self,Z):
        """
        This function will be used by the output layer of the net for computing final predictions
        :param Z:  The Z output of the current layer in the net
        :return:  returns the value sigmoid(Z)
        """
        return 1/(1+np.exp(-Z))

    def forward_prop(self,x):
        """
        computes forward propagation for a given dataset x of the shape (n_x,m)
        where n_x is the number of features per training examples
        and m is the total number of training examples
        """
        self.Z1=np.dot(self.W1,x)+ self.b1
        self.A1=np.tanh(self.Z1)
        self.Z2=np.dot(self.W2,self.A1)+self.b2
        self.A2=self.sigmoid(self.Z2)

    def compute_cost(self):
        """
        Calculates the cost of the model which will be used in backward propagation
        """
        b=np.log(self.A2)
        c=np.log(1-self.A2)
        self.cost=-(np.dot(self.y,b.T)+np.dot(1-self.y,c.T))/self.m
        self.cost=float(np.squeeze(self.cost))

    def backward_prop(self):
        """ This computes backward propagation"""
        g1=1-np.power(self.A1,2)
        self.dZ2=self.A2-self.y
        self.dW2=np.dot(self.dZ2,self.A1.T)/self.m
        self.db2=np.sum(self.dZ2,axis=1,keepdims=True)/self.m
        self.dZ1=np.dot(self.W2.T,self.dZ2)*g1
        self.dW1=np.dot(self.dZ1,self.x.T)/self.m
        self.db1=np.sum(self.dZ1,axis=1,keepdims=True)/self.m

    def update_params(self):
        """ To carry gradient descent and update the parameters"""
        self.W1=self.W1-(self.rate*self.dW1)
        self.b1=self.b1-(self.rate*self.db1)
        self.W2=self.W2-(self.rate*self.dW2)
        self.b2=self.b2-(self.rate*self.db2)

    def learn(self,print_cost=False,per_itr=1000):
        """ performs forward prop, then computes cost, performs backward prop and then updates the params for the
        specified number of iterations. Also, prints cost every 1000 iterations if print_cost is True
        :param per_itr : after how many iterations cost is to be printed
        """

        for i in range(self.itr):
            self.forward_prop(self.x)
            self.compute_cost()
            self.backward_prop()
            self.update_params()

            if i%per_itr==0:
                self.costs.append(self.cost)

            if print_cost and i%per_itr==0:
                print('Cost after iteration {} : {}'.format(i,self.cost))

        self.accuracy_train=self.accuracy(self.x,self.y)
        self.accuracy_test=self.accuracy(self.x_test,self.y_test)
        print('------Training Set Accuracy : {}%------'.format(self.accuracy_train))
        print('------Testing Set Accuracy  : {}%------'.format(self.accuracy_test))


    def predict(self,x):
        """
        :param x: dataset for which outputs are to be predicted. Shape- (n_x,m)
        :return: predicted outputs for given dataset,array of 0 and 1
        """
        m=x.shape[1]
        y_predictions=np.zeros((1,m))
        self.forward_prop(x)
        y_predictions=np.where(self.A2>0.5,1,0)

        return y_predictions

    def accuracy(self,x,y):
        """
        :param x: dataset of the form (n_x,m) for which accuracy is to be tested
        :param y: output array for given dataset x of the form (1,m) or (n_y,1,m)
        :return: returns the accuracy for the predictions made by the model as compared to y
        """
        m=x.shape[1]
        temp=self.predict(x)
        temp=np.abs(temp-y)
        temp=float(np.squeeze(np.sum(temp)))
        temp=100-((temp/m)*100)
        return temp

    def plot_decision_bound(self,x, y,step):
        """
        Plots the decision boundary based on the current values of parameters of the hidden and output layer
        :param x: the numpy array representing the x coordinates
        :param y: the numpy array representing the y coordinates
        :param step: the difference between the individual points to be used for generating the grid
        """
        min1, min2 = x.min(), y.min()  # get the minimum values for both x and y
        max1, max2 = x.max(), y.max()  # get the maximum values for both x and y
        maxim = max(max1, max2) + 1    # calculate the maximum of all
        minim = min(min1, min2) - 1    # calculate the minimum of all
        grid = np.arange(minim, maxim, step)  # generate a grid array with step as distance between each value generated
        xx, yy = np.meshgrid(grid, grid)    # get the grid arrays representational of the whole grid
        x = np.zeros((2, xx.flatten().shape[0]))
        x[0, :] = xx.flatten()
        x[1, :] = yy.flatten()
        y=self.predict(x)
        plt.scatter(x[0,:],x[1,:],c=y,cmap=plt.cm.Spectral)  # plot the decision boundary

    def plot_learning_curve(self):
        """
        Plots the learning curve for the model with x axis as every 1000 iterations and y axis as the corresponding
        value of the cost at the specified iteration
        Use the command ----plt.show()---- to be able to see the curve after calling this function
        """
        costs=np.squeeze(self.costs)
        plt.plot(costs,label='{}({},{})'.format(self.rate,self.accuracy_train,self.accuracy_test))
        plt.ylabel('Cost')
        plt.xlabel('Iterations(per thousands)')
        plt.legend()
        plt.title('----Shallow net with one hidden layer----\n----Legend----[ Rate(train accuracy,test accuracy) ] ')

class deep_net():
    def __init__(self,x_train,y_train,x_test,y_test,layer_dims,learning_rate,iterations,lambd,keep_prob):
        """
        :param x_train: the training dataset on which the model will learn
                        should be of the form,(n_x,m) where n_x is the number of features in each training example
                        and m is the total number of training examples
        :param y_train: the output for the training data set. Should be of the shape (1,m) or (n_y,1,m) with values 1 or 0
                        ,where n_y will be the number of nodes in output layer
        :param x_test:  the testing data set, should be of the form (n_x,m)
        :param y_test:  the output for the testing data set. Should be of the shape (1,m) or (n_y,1,m) with values 1 or 0
        :param layer_dims: A list containing the dimensions(no. of nodes) for each layer in the neural net
        :param learning_rate: the rate at which params will be updated(gradient descent)
        :param iterations: total number of iterations over the training set for which the model will be learning
        """
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.rate=learning_rate
        self.itrs=iterations
        self.L=len(layer_dims)
        self.layer_dims = layer_dims
        self.lambd=lambd
        self.keep_prob=keep_prob
        self.parameters={}              # This dict will be used for storing parameters
        self.v={}
        self.s={}
        np.random.seed(3)
        for l in range(1,self.L):            # Initializing the parameters using He initialization
            self.parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
            self.parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

            self.v['dW'+str(l)] = np.zeros(( self.parameters['W'+str(l)].shape ))
            self.v['db' + str(l)] = np.zeros((self.parameters['b' + str(l)].shape))

            self.s['dW' + str(l)] = np.zeros((self.parameters['W' + str(l)].shape))
            self.s['db' + str(l)] = np.zeros((self.parameters['b' + str(l)].shape))

    def random_mini_batches(self, X , Y , mini_batch_size = 64 ,seed =0):
        np.random.seed(seed)
        m=X.shape[1]
        mini_batches=[]

        #Shuffle X and Y:
        permutation= list(np.random.permutation(m))
        shuffled_X=X[:,permutation]
        shuffled_Y=Y[:,permutation].reshape((1,m))

        #Partitioning X and Y:
        num_complete_mini_batches=math.floor(m/mini_batch_size)
        for k in range(0,num_complete_mini_batches):
            mini_batch_X = shuffled_X[: , k*mini_batch_size: ((k*mini_batch_size)+mini_batch_size) ]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: ((k * mini_batch_size) + mini_batch_size)]
            mini_batch = ( mini_batch_X , mini_batch_Y )
            mini_batches.append( mini_batch )

        # Handling the end case ( last batch size < mini_batch_size )
        if m%mini_batch_size != 0:
            mini_batch_X = shuffled_X[: , num_complete_mini_batches * mini_batch_size : m ]
            mini_batch_Y = shuffled_Y[:, num_complete_mini_batches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def sigmoid(self,Z):
        """computes activation,returns activation as well as input Z as cache for backprop"""
        A=1/(1+np.exp(-Z))
        return A,Z

    def relu(self,Z):
        """computes activation,returns activation as well as input Z as cache for backprop"""
        A=np.maximum(0,Z)
        return A,Z

    def sigmoid_backward(self,dA,activation_cache):
        dg,Z=self.sigmoid(activation_cache)
        dZ=dA*dg*(1-dg)
        return dZ

    def relu_backward(self,dA,activation_cache):
        dg,Z=self.relu(activation_cache)
        dZ=np.array(dA,copy=True)  #just converting dZ to a correct object
        dZ[Z<=0]=0
        return dZ

    def linear_forward(self,A,W,b):
        """
        Computes linear forward i.e Z=W.A+b for one layer of the neural net
        :param A: The activation received from the previous layer
        :param W: Weight matrix of the current layer
        :param b: bias vector for the current layer
        :return:1.)Z - The input for the activation for the next layer
                2.)cache- A tuple consisting of A,W and b which will be used for back prop later
        """
        Z= np.dot(W,A)+b
        cache=(A,W,b)

        return Z,cache

    def linear_activation_forward(self,A_prev,W,b,activation):
        """
        Computes the activation for a given layer l
        :param A_prev: The activation from the previous layer
        :param W: The weight matrix of the current layer
        :param b: The bias vector of the current layer
        :param activation: The activation function to be used. Either - 'sigmoid' or 'relu'
        :return: 1.)A-Activation output of the layer
                 2.)cache- A python tuple consisting of linear cache(A_prev,W,b)and activation cache(Z)
        """
        if activation == 'sigmoid':
            Z,linear_cache=self.linear_forward(A_prev,W,b)
            A,activation_cache=self.sigmoid(Z)

        elif activation == 'relu':
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A, activation_cache = self.relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(self,x,parameters):
        """
        Computes forward propagation for the L-layer model
        :param x: input vector for the neural net
        :return:
        """
        caches=[]             #cache storing linear and activation cache for all layers, will be used in backprop
        self.dropout_caches={}  #cache storing the values for dropout matrices for all layers except input and
                                # output layer. Total length of this python dictionary will be (L-1) where L is the
                                # total number of layers in the neural net
        L=len(parameters)//2  # number of layers in the neural_net
        A=x     #activation for first layer

        for l in range(1,L):   #iterating over all the layers except the output layer
            A_prev=A
            W,b=parameters['W'+str(l)],parameters['b'+str(l)]
            A,cache=self.linear_activation_forward(A_prev,W,b,'relu')
                                                                   # Applying Inverted Dropout
            D = np.random.rand(A.shape[0], A.shape[1])          # Step 1: initialize matrix D for dropout
            D = (D < self.keep_prob).astype('int')                    # Step 2: convert entries of D to 0 or 1
            A = np.multiply(A, D)                               # Step 3: shut down some neurons of A
            A = A / self.keep_prob                     # Step 4: scale the value of neurons that haven't been shut down
            self.dropout_caches['D'+str(l)]=D
            caches.append(cache)

        A_prev=A                    #computing linear activation forward for the output layer
        AL,cache=self.linear_activation_forward(A_prev,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
        caches.append(cache)

        return AL, caches

    def compute_cost(self,AL,YL):   #computes cost function
        Y=YL
        A=AL
        m=Y.shape[1]
        #cost=-(np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T))
        logprobs = np.multiply(-np.log(A), Y) + np.multiply(-np.log(1 - A), 1 - Y)
        cost=(1./m)*np.nansum(logprobs)
        cost=np.squeeze(cost)

        return cost

    def compute_cost_with_regularization(self,AL,YL):
        m=YL.shape[1]
        Frobenius_norm=0

        for l in range(1,self.L):
            W=self.parameters['W'+str(l)]
            Frobenius_norm+=np.sum(np.square(W))

        L2_cost=(self.lambd/(2*m))*Frobenius_norm
        cross_entropy_cost=self.compute_cost(AL,YL)
        cost= cross_entropy_cost+ L2_cost
        cost = np.squeeze(cost)

        return cost

    def linear_backward(self,dZ,cache):
        """
        Computes back prop for one layer
        :param dZ: input from one layer ahead
        :param cache: cache contains the values A_prev,W,b for the current layer
        :returns: 1.)dA_prev-input for previous layer's back prop
                  2.)dW- adjustment for weights of current layer
                  3.)db- adjustment for bias vectors of current layer
        """
        A_prev,W,b=cache
        m=A_prev.shape[1]

        dW=np.dot(dZ,A_prev.T)/m + (self.lambd/m)*W
        db=np.sum(dZ,axis=1,keepdims=True)/m
        dA_prev=np.dot(W.T,dZ)

        return dA_prev,dW,db

    def linear_activation_backward(self,dA,cache,activation):
        """
        computes back prop for layer l
        :param dA: input from the layer ahead
        :param cache: tuple (linear_cache,activation_cache). linear_cache -(A+prev,W,b) Activation_cache -(Z)
        :param activation: whether 'sigmoid' or 'relu'
        """
        linear_cache,activation_cache=cache

        if activation=='relu':
            dZ=self.relu_backward(dA,activation_cache)
            dA_prev,dW,db=self.linear_backward(dZ,linear_cache)

        elif activation=='sigmoid':
            dZ=self.sigmoid_backward(dA,activation_cache)
            dA_prev,dW,db=self.linear_backward(dZ,linear_cache)

        return dA_prev,dW,db


    def backward_prop(self,AL,YL,caches):   #computes backward propagation
        grads={}   #for storing gradient descent values
        L=len(caches)  #number of layers
        m=AL.shape[1]  # no. of training examples
        Y=YL.reshape(AL.shape)

        #dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))  # Initializing back prop

        current_cache=caches[L-1]   # for last layer
        #grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)]=self.linear_activation_backward(dAL,current_cache,'sigmoid')

        lin_cache, _ = current_cache
        dZL = AL - Y
        grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = self.linear_backward(dZL, lin_cache)


        # Implementing back propagation part of inverted dropout for output layer feeding into its previous layer
        grads['dA' + str(L - 1)] = np.multiply(grads['dA' + str(L - 1)], self.dropout_caches['D' + str(L-1)])
        # Applying mask D
        grads['dA' + str(L - 1)] = grads['dA' + str(L - 1)] / self.keep_prob  # Scaling the values

        for l in reversed(range(L-1)):  #iterating over remaining layers
            current_cache=caches[l]
            dA_prev_temp,dW_temp,db_temp=self.linear_activation_backward(grads['dA'+str(l+1)],current_cache,'relu')

            if l!=0:
                # Implementing back propagation part of inverted dropout for remaining layers
                dA_prev_temp = np.multiply(dA_prev_temp,self.dropout_caches['D'+str(l)])  # Applying mask D
                dA_prev_temp = dA_prev_temp/ self.keep_prob                               # Scaling the values

            grads['dA'+str(l)]=dA_prev_temp
            grads['dW'+str(l+1)]=dW_temp
            grads['db'+str(l+1)]=db_temp

        return grads

    def update_params(self,parameters,grads,rate):   #gradient descent
        L=len(parameters)//2

        for l in range(L):
            parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-(rate*grads['dW'+str(l+1)])
            parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - (rate * grads['db' + str(l + 1)])

        return parameters

    def update_parameters_with_momentum(self,parameters,grads,v,beta,rate):
        L = len(parameters) // 2

        for l in range(L):
            # Updating velocities :
            v['dW'+str(l+1)] = beta * v['dW' + str(l+1)] + (1-beta) * grads['dW' + str(l+1)]
            v['db' + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

            # Updating parameters :
            parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - (rate * v['dW' + str(l + 1)])
            parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - (rate * v['db' + str(l + 1)])

        return parameters , v

    def update_parameters_with_adam(self,parameters,grads,v,s,t, rate, beta1=0.9,beta2=0.999,epsilon= 1e-8,):
        L = len(parameters) // 2
        v_corrected = {}
        s_corrected = {}

        for l in range(L):
            # Updating velocities :
            v['dW'+str(l+1)] = beta1 * v['dW' + str(l+1)] + (1-beta1) * grads['dW' + str(l+1)]
            v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

            s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * np.square(grads['dW' + str(l + 1)])
            s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * np.square(grads['db' + str(l + 1)])

            # Implementing bias correction :
            v_corrected['dW'+ str(l+1)] = np.divide( v['dW' + str(l+1)] , (1 - beta1**t) )
            v_corrected['db' + str(l + 1)] = np.divide(v['db' + str(l + 1)], (1 - beta1 ** t))

            s_corrected['dW' + str(l + 1)] = np.divide(s['dW' + str(l + 1)], (1 - beta2 ** t))
            s_corrected['db' + str(l + 1)] = np.divide(s['db' + str(l + 1)], (1 - beta2 ** t))

            # Updating parameters :
            parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - (rate * np.divide( v_corrected['dW' + str(l + 1)] , np.sqrt( s_corrected['dW' + str(l+1)] ) + epsilon  ) )
            parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - (
                        rate * np.divide(v_corrected['db' + str(l + 1)],
                                         np.sqrt(s_corrected['db' + str(l + 1)]) + epsilon))

        return parameters , v , s

    def learn(self,print_cost=False,per_itrs=100):     #lerning step for the neural_net
        np.random.seed(1)
        self.per_itrs=per_itrs
        self.costs=[]
        self.dev_errors=[]
        self.train_errors=[]
        self.plot_parameters={}

        for i in range(self.itrs):
            self.A,self.caches=self.forward_prop(self.x_train,self.parameters)
            self.cost=self.compute_cost_with_regularization(self.A,self.y_train)
            self.grads=self.backward_prop(self.A,self.y_train,self.caches)
            self.parameters=self.update_params(self.parameters,self.grads,self.rate)

            if i%per_itrs==0:
                self.costs.append(self.cost)
                self.dev_errors.append(self.accuracy(self.x_test, self.y_test))
                self.train_errors.append(self.accuracy(self.x_train, self.y_train))

            if print_cost and i%per_itrs==0:
                print('Cost after epoch {} : {}'.format(i,self.cost))

        self.plot_parameters['Costs']=self.costs
        self.train_errors=(100-np.array(self.train_errors))/100
        self.dev_errors = (100 - np.array(self.dev_errors)) / 100
        self.plot_parameters['Training Error']=self.train_errors
        self.plot_parameters['Dev Error']=self.dev_errors

        self.accuracy_train = self.accuracy(self.x_train, self.y_train)
        self.accuracy_test = self.accuracy(self.x_test, self.y_test)
        print('------Training Set Accuracy : {}%------'.format(self.accuracy_train))
        print('------Testing Set Accuracy  : {}%------'.format(self.accuracy_test))

    def learn_with_optimization(self, optimizer, mini_batch_size, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                num_epochs=10000, print_cost=False,
                                per_itrs=100):  # lerning step with optimization for the neural_net

        self.per_itrs = per_itrs
        self.costs=[]
        t=0
        seed =10
        m=self.x_train.shape[1]
        self.dev_errors=[]
        self.train_errors=[]
        self.plot_parameters={}

        for i in range(num_epochs):

            seed = seed + 1
            mini_batches = self.random_mini_batches(self.x_train, self.y_train, mini_batch_size , seed)
            cost_total = 0

            for mini_batch in mini_batches:

                # Select a mini_batch :
                ( mini_batch_X , mini_batch_Y ) = mini_batch

                self.A,self.caches=self.forward_prop(mini_batch_X ,self.parameters)
                cost_total += self.compute_cost_with_regularization(self.A, mini_batch_Y) * mini_batch_Y.shape[1]
                self.grads=self.backward_prop(self.A, mini_batch_Y ,self.caches)

                # Update parameters :
                if optimizer == 'gd':
                    self.parameters=self.update_params(self.parameters,self.grads,self.rate)
                elif optimizer == 'momentum':
                    self.parameters , self.v = self.update_parameters_with_momentum(self.parameters , self.grads,
                                                                                    self.v , beta , self.rate)
                elif optimizer == 'adam':
                    t= t + 1 # current mini_batch number for adam
                    self.parameters, self.v ,self.s = self.update_parameters_with_adam(self.parameters, self.grads,
                                                                                       self.v, self.s , t , self.rate ,
                                                                                       beta1 , beta2, epsilon)

            self.cost = cost_total / m
            if i%per_itrs==0:
                self.costs.append(self.cost)
                self.dev_errors.append(self.accuracy(self.x_test, self.y_test))
                self.train_errors.append(self.accuracy(self.x_train, self.y_train))

            if print_cost and i%per_itrs==0:
                print('Cost after epoch {} : {}'.format(i,self.cost))

        self.plot_parameters['Costs']=self.costs
        self.train_errors=(100-np.array(self.train_errors))/100
        self.dev_errors = (100 - np.array(self.dev_errors)) / 100
        self.plot_parameters['Training Error']=self.train_errors
        self.plot_parameters['Dev Error']=self.dev_errors

        self.accuracy_train = self.accuracy(self.x_train, self.y_train)
        self.accuracy_test = self.accuracy(self.x_test, self.y_test)
        print('------Training Set Accuracy : {}%------'.format(self.accuracy_train))
        print('------Testing Set Accuracy  : {}%------'.format(self.accuracy_test))


    def predict(self,x):
        """
        :param x: dataset for which outputs are to be predicted. Shape- (n_x,m)
        :return: predicted outputs for given dataset,array of 0 and 1
        """
        m=x.shape[1]
        y_predictions=np.zeros((1,m))
        self.A,self.caches=self.forward_prop(x,self.parameters)
        y_predictions=np.where(self.A>0.5,1,0)

        return y_predictions

    def accuracy(self,x,y):
        """
        :param x: dataset of the form (n_x,m) for which accuracy is to be tested
        :param y: output array for given dataset x of the form (1,m) or (n_y,1,m)
        :return: returns the accuracy for the predictions made by the model as compared to y
        """
        m=x.shape[1]
        temp=self.predict(x)
        temp=np.abs(temp-y)
        temp=float(np.squeeze(np.sum(temp)))
        temp=100-((temp/m)*100)
        return temp

    def plot_learning_curve(self,plot_parameters):
        """
        Plots the learning curve for the model with x axis as every 1000 iterations and y axis as the corresponding
        value of the cost at the specified iteration
        Use the command ----plt.show()---- to be able to see the curve after calling this function
        """
        for param in plot_parameters:
            costs=np.squeeze(plot_parameters[param])
            plt.plot(costs,label='{}_{}({},{})'.format(param,self.rate,self.accuracy_train,self.accuracy_test))
        plt.ylabel('Values')
        plt.xlabel('Epoch Number (X{})'.format(self.per_itrs))
        plt.legend()
        plt.title('----Deep net with {} layers----\n----Legend----[ Rate(train accuracy,test accuracy) ] '.format(self.L-1))

    def plot_decision_bound(self,x, y, color, step):
        """
        Plots the decision boundary based on the current values of parameters of the hidden and output layer.
        To be used for plotting 2-D data only
        :param x: the numpy array representing the x coordinates
        :param y: the numpy array representing the y coordinates
        :param step: the difference between the individual points to be used for generating the grid
        :param color :  the numpy array containing the color info for each datapoint, contains the values zeros or ones
        """
        x_orig, y_orig = x, y
        min1, min2 = x.min(), y.min()  # get the minimum values for both x and y
        max1, max2 = x.max(), y.max()  # get the maximum values for both x and y
        maxim = max(max1, max2) #+ 1     calculate the maximum of all
        minim = min(min1, min2) #- 1     calculate the minimum of all
        grid = np.arange(minim, maxim, step)  # generate a grid array with step as distance between each value generated
        xx, yy = np.meshgrid(grid, grid)    # get the grid arrays representational of the whole grid
        x = np.zeros((2, xx.flatten().shape[0]))
        x[0, :] = xx.flatten()
        x[1, :] = yy.flatten()
        y=self.predict(x)
        plt.scatter(x[0,:],x[1,:],c=y,cmap=plt.cm.Spectral)  # plot the decision boundary
        plt.scatter(x_orig, y_orig, c=color, cmap='gray')
        plt.scatter(x_orig, y_orig, c=color, cmap=plt.cm.Spectral)

    def gradient_check(self, x , y , parameters , epsilon = 1e-7):
        gradients = {}

        A, caches = self.forward_prop(x, parameters)
        cost = self.compute_cost(A, y)
        grads = self.backward_prop(A, y, caches)

        parameters_values = dictionary_to_vector(parameters)

        for l in range(1,self.L):
            gradients['dW'+str(l)]= grads['dW'+str(l)]
            gradients['db' + str(l)] = grads['db' + str(l)]

        grad = dictionary_to_vector(gradients)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters,1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        #Compute gradapprox :
        for i in range(num_parameters):

            thetaplus = np.copy(parameters_values)
            thetaplus[i][0] += epsilon
            A, _ = self.forward_prop(x, vector_to_dictionary(thetaplus,self.layer_dims))
            J_plus[i] = self.compute_cost(A, y)

            thetaminus = np.copy(parameters_values)
            thetaminus[i][0] -= epsilon
            A, _ = self.forward_prop(x, vector_to_dictionary(thetaminus, self.layer_dims))
            J_minus[i] = self.compute_cost(A, y)

            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator/denominator

        if difference > 2e-7 :
            print('----There is a mistake in backward propagation :( ----\n----Difference = {} ----'.format(difference))
        else:
            print('----Backward propagation works perfectly fine :) ----\n----Difference = {} ----'.format(difference))

        return difference

x_train=load_data('datasets/train_catvnoncat.h5','train_set_x')
x_train=standardize_image(x_train)
y_train=load_data('datasets/train_catvnoncat.h5','train_set_y')
y_train=y_train.reshape((1,y_train.shape[0]))
x_test=load_data('datasets/test_catvnoncat.h5','test_set_x')
x_test=standardize_image(x_test)
y_test=load_data('datasets/test_catvnoncat.h5','test_set_y')
y_test=y_test.reshape((1,y_test.shape[0]))

model=deep_net(x_train,y_train,x_test,y_test,[12288,20,30,1],0.03,1300,0,1)
#model=Logistic_Regression(x_train,y_train,x_test,y_test,0.0007,2000)
#model.learn(print_cost=True)

#model.learn(print_cost=True,per_itrs=100)