import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import Sequential, optimizers
from keras.layers import Dropout, Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import plot_model


class NeuralNetwork:
    model = ""
    history = ""
    questions = []
    answers = []
    nbUsedLines = 5000
    
    def __init__(self): 
        self.nom = 'coucou'
    
    def loadData(self, path):
        print("Loading data ...")
        
        nbLine = 0
        
        with open(path) as dataset:
            for line in dataset:
                lineList = line.split(';')

                question = np.fromstring(lineList[0][3:-2], dtype=float, sep=',')*10
                answer = np.fromstring(lineList[1][3:-4], dtype=float, sep=',')*10

                self.questions.append(question)
                self.answers.append(answer)
        
                if nbLine < self.nbUsedLines-1:
                    nbLine += 1
                else:
                    break
        
        #self.questions = [[0,0], [1,1], [1,0], [0,1]]
        #self.answers = [[0], [0], [1], [1]]
       
       
        self.questions = np.array(self.questions)
        self.answers = np.array(self.answers)

        if len(self.questions) == len(self.answers):
            print("Done ! " + str(len(self.questions)) + " Q/A loaded.")
        else:
            print("An error occured, please check if the number of Q/A is the same.")

    def createNNStructure(self):
        print("Creating NN structure ...")
        
        self.model = Sequential()
        self.model.add(Dense(370, input_shape=(100,) ,activation="tanh"))
        self.model.add(Dense(400, input_shape=(100,) ,activation="relu"))
        self.model.add(Dense(370, input_shape=(100,) ,activation="tanh"))
        self.model.add(Dense(100, activation="tanh"))
        #sgd = optimizers.SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['accuracy'])
        self.model.summary()
        
        """ 
        self.questions = self.questions.reshape(self.nbUsedLines, 1, 100)
        self.answers = self.answers.reshape(self.nbUsedLines, 1, 100)
        self.model = Sequential()
        self.model.add(LSTM(units=100, input_shape=(1,100), return_sequences=True))
        self.model.add(Dense(100, activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        self.model.summary()
        """

        print("Done !")

    def train(self):
        print("Training starting !")
        self.history = self.model.fit(self.questions, self.answers, epochs=701, batch_size=16)
        print("Training done !")

    def displayStats(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()

        # Plot training & validation accuracy values
        plt.plot(self.history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()


print(' ')
print(' ')
print(' ')
NN = NeuralNetwork()
NN.loadData("data_nlp_cornell_new.txt")
NN.createNNStructure()
NN.train()
NN.displayStats()
