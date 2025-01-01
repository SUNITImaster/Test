"""
1. This code is designed to test Weirstrass Approximation theorem..ie. Polynomials are dense in C[a,b]
2. We first export data from hand drawn graph using plotdigitizer app and then import this x-y csv data into
python and plot
3. Once we see the plot, we can try fitting Bernstein polynomial for large N and see if the error decreases with increasing N
4. Also study the graph of fitted Bernstein Polynomial
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from math import comb

class Bernstein_polynomial():
    
    def __init__(self,degree):
        self.degree=degree
        self.func_arr=np.ndarray(shape=(((degree+1),1)),dtype=np.float16)
        

    def eval_graph(self,x_val,y_val):
        sum=np.zeros_like(x_val)
        idx=0
        for x in x_val:
            for exp in range (0 ,(self.degree+1),1):
                #find x closest to exp/degree
                k_by_n=np.float16(exp/self.degree)
                idx_y=np.argmin(np.abs((x_val-k_by_n)))
                #print(idx_y)
                #print(k_by_n)
                #print(x_val[idx_y],y_val[idx_y])
                sum[idx] = sum[idx] + ((comb(self.degree,exp))* np.power(x_val[idx],exp)*np.power((1-x_val[idx]),(self.degree-exp))*y_val[idx_y])
                #bernstein_array=np.ndarray(shape=(((degree+1),1)),dtype=np.float16)
            
            
            idx=idx+1
            

        return sum

    

if __name__=="__main__":

    print("started the main code of Weirstrass Approximation theorm")
    N=1001 # This is the max value uptil which we iterate
    os.chdir("D:/Suniti/GitPythonRepo/NNetUAT/Polynomial/Weirstrass_approximation_theorem/")
    hand_graph_data=pd.read_csv("hand_plot_graph_data.csv") #Underlying data of an arbitrary continuous hand plotted graph extracted into csv"
    hand_graph_data.columns=["X_value","Y_value"]
    hand_graph_data["X_value_norm"]=hand_graph_data["X_value"]/11
    l=len(hand_graph_data["X_value"])
    print(hand_graph_data.head(n=2))
    x_val_norm=hand_graph_data["X_value_norm"]
    y_val=hand_graph_data["Y_value"]
    plt.scatter(x_val_norm,y_val,s=3,color="black")
    #plt.show()
    print(comb(10,3))
    #Bernstein_array=np.ndarray(shape=(11,l,2))
    
    graph_array_list=[]
    colorlist=["red","green","orange","violet","yellow"]
    for n in range(200,N,200):
        b_n = Bernstein_polynomial(degree=n)
        Bernstein_array=b_n.eval_graph(x_val_norm,y_val) # this will evaluate Bn polynomial of degree n and save it in array
        plt.scatter(x_val_norm,Bernstein_array,s=3,color=colorlist.pop(0))
        graph_array_list.append((Bernstein_array,n))
        print("done for "+str(n))
    
    with open("array_value.txt", 'w') as output:
        for row in graph_array_list:
            output.write(str(row) + '\n')
   

    plt.legend(["orig_func","200","400","600","800","1000"])
    plt.show()

