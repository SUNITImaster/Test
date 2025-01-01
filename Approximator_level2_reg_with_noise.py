"""
Data generator Module

# Generate data for say y = X2 + Sin(x) + exp(x2) + noise 

end here

Neuralnet Module

## have a simple 1-2 layers NNet and test different kind of activation functions. 


End NNEt Module


Main code snippet

Initialize data generator class
call the class.generatedata(x1,x2,target or y) which returns the full dataset
Call the batch generator to divide the data into train & test and in batches for NNet training
Initialize the nnet module.
Call the train method of the NNET class
Call the print module to print out the learned weights.

"""

import torch 
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader,TensorDataset, random_split,RandomSampler
from torch.nn.functional import cross_entropy,mse_loss
import os
from sklearn.metrics import confusion_matrix

class DataGenerator():

    def __init__(self,total_size):
        self.total_size=total_size
        self.batch_size = 256
        self.X=torch.rand(size=[self.total_size],dtype = torch.float32)*200

    def getData(self):
        ##create Y=x2+3sin(x)+exp(x2)+noise
       
        self.Y=torch.zeros_like(self.X,dtype=torch.float32)
        
        self.Y=torch.sqrt(self.X)+3*torch.sin(self.X)
        data_club= TensorDataset(self.X,self.Y)
        return data_club
    
    def batch_data(self,x):
        rsampler=RandomSampler(x)
        train_data,test_data,val_data=random_split(x,lengths=[60000,20000,20000])
        train_datagen = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_datagen = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        val_datagen = DataLoader(val_data, batch_size=10000, shuffle=True)
        return train_datagen,test_datagen,val_datagen

    def printplots(self):
        X_val=self.X.detach().numpy()
        Y_val=self.Y.detach().numpy()
        x_ax=X_val.squeeze()
        y_ax=Y_val.squeeze()
        print(x_ax.shape)
        print(y_ax.shape)
        sp=seaborn.scatterplot(x=x_ax,y=y_ax)
        plt.show()

class Neural_Net(nn.Module):
    def __init__(self):
        super(Neural_Net,self).__init__()
        self.num_epochs = 500
        self.learning_rate = 0.0001
        self.record_freq = 20
        self.output_path = "results_level2/"
        self.model_output = self.output_path + "modelweights/"
        self.train_loss=[]
        self.test_loss=[]
        self.val_loss=[]


        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists(self.model_output):
            os.makedirs(self.model_output)
        
        self.nnet_model=nn.Sequential(OrderedDict([
         ('Linear_l1',nn.Linear(in_features=1,out_features=512)),
         #('Relu1',nn.ReLU()),
         ('Tanh1',nn.Tanh()),
         ('Linear_l2',nn.Linear(in_features=512,out_features=1024)),
         #('Relu2',nn.ReLU()),
         ('Tanh2',nn.Tanh()),
        ('Linear_l3',nn.Linear(in_features=1024,out_features=2048)),
         #('Relu2',nn.ReLU()),
         ('Tanh3',nn.Tanh()),
         ('Linear_l4',nn.Linear(in_features=2048,out_features=1))
         
        ]))

    def forward(self,input):
        output=self.nnet_model(input.reshape([len(input),1]))
        return output

    def TrainNetwork(self,train_datagen,test_datagen,val_datagen):

        optimizer=Adam(self.nnet_model.parameters(),lr=self.learning_rate)
        
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
        
            running_loss = 0.0

            for batch_idx, data in enumerate(train_datagen):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss=mse_loss(outputs.squeeze(),labels,reduction='mean')
                running_loss=running_loss+loss.item()
                loss.backward()
                optimizer.step()

            running_test_loss = 0.0

            for batch_idx_t, data_t in enumerate(test_datagen):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data_t
                outputs = self.forward(inputs)

                loss_t=mse_loss(outputs.squeeze(),labels,reduction='mean')
                running_test_loss=running_test_loss+loss_t.item()


   
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss per batch: ", running_loss / (batch_idx))
            print("\tAvgLoss for test is ",running_test_loss/batch_idx)

            
            self.train_loss.append((epoch,(running_loss/batch_idx)))
            self.test_loss.append((epoch,(running_test_loss/batch_idx_t)))
 
            
        
            if(epoch%self.record_freq==0):
                filenamepath=self.model_output+"model_optim_epoch_"+str(epoch)+".pt"
                torch.save(
                    {'epoch':epoch,
                     'model_dict':self.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss':running_loss
                     }, filenamepath
                )
        
        return None
    


if __name__=="__main__":

    print("started the main code")
    os.chdir("D:/Suniti/GitPythonRepo/NNetUAT")
    torch.set_printoptions(edgeitems=5)
    total_size=100000
    datagen=DataGenerator(total_size)
    data_club= datagen.getData()
    data_club_train,data_club_test,data_club_val=datagen.batch_data(data_club)
    datagen.printplots()
    model=Neural_Net()
    #loaded_dict=torch.load("D:/Suniti/GitPythonRepo/NNetUAT/results_level2/modelweights/model_optim_epoch_300.pt",weights_only=False)
    #model.load_state_dict(state_dict=loaded_dict["model_dict"])
    model.TrainNetwork(data_club_train,data_club_test,data_club_val)
    for batch_idx, dataval in enumerate(data_club_val):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = dataval 
                output=model.forward(inputs)
                mse= mse_loss(output.squeeze(),labels,reduction='mean')
                print(mse)
                sp=seaborn.scatterplot(x=inputs.detach().numpy().squeeze(),y=output.detach().numpy().squeeze())
                plt.show()







