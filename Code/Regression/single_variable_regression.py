import traceback
import pylab
import numpy
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from numpy import arange,array,ones,linalg
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from pylab import plot,show



def loadData(filename):
    try:
        print filename
        data=numpy.loadtxt(filename,delimiter=",")
        return data
    except:
        print("Exception occured: Please check Filename or File path"); 

def initializeParameter(filedata):
    # setting parameter for model
    m=filedata[:, 1].size #Number of rows
    iterations = 10 #number Of iteration (Value Depends on you)
    alpha = 0.01 #Learning Rate (should be between 0-1)
    Q0=6        #Initial value of intercept (Value Depends on you)
    Q1=1.5      #Initial value of slope (Value Depends on you)
    parameters={"m":m,"iterations":iterations,"alpha":alpha,"Q0":Q0,"Q1":Q1}
    return parameters;

def visualizeData(filedata,parameters):
    Q0=parameters["Q0"]
    Q1=parameters["Q1"]
    x=filedata[:, 0]#0 for get first column
    y=filedata[:, 1]#1 for get second column

    title(' distribution')
    xlabel('X')
    ylabel('Y')
    line1=Q1*x+Q0 #Line Equation Y=MX+C M(Here Q1) is slope,C(Here Q0) is intercept
    plot(x,line1,'r-',x,y,'o')
    show()#For popup graph 

# cast calculation
def cost(x,y,q0,q1,datasize):
    tempsum=0
    for i in range(1,datasize):
        #Cost Funtion name:Square root ,Y^=q0+(x[i]*q1)
        sqErrors =((q0+(x[i]*q1))-y[i])
        tempsum=tempsum+(sqErrors **2)
    totalCost = (1.0 / (2 * datasize))*tempsum
    return totalCost

# Gradient descent algo
def gradient_descent(x,y,q0,q1,parameters):
    alpha=parameters["alpha"]
    for i in range(1,parameters["iterations"]):
        D_DQ=jcastfornew_Q(x,y,q0,q1,parameters)      
        q1=q1-(alpha*D_DQ[0])
        q0=q0-(alpha*D_DQ[1])
        line=q0+q1*x
        plot(x,line,'r-',x,y,'o')
        show()
    finalvalue =[]
    finalvalue.append(q0)
    finalvalue.append(q1)
    return  finalvalue #final Q0 Q1 value return here

#j(Q0,Q1) calculation
def jcastfornew_Q(x,y,Q0,Q1,parameters):                   #calculation for slop j(Q0,Q1)
    temp1=0
    temp2=0
    m=parameters["m"]
    for i in range(1,m):  
        temp1=temp1+(((Q0+x[i]*Q1)-y[i])*x[i])  #for Q1
        temp2=temp2+(((Q0+x[i]*Q1)-y[i]))       #for Q0
    temp1=temp1/m
    temp2=temp2/m
    Slops =[]
    Slops.append(temp1)
    Slops.append(temp2)
    return Slops


#controler block code 
try:
    #step 1) Load data from file
    filedata=loadData('./../../Dataset/mydataset.txt')

    #filedata=loadData('/home/aakash/Documents/CodeWork/Learning/MachineLearning/Machine_Learning_Course/Dataset/mydataset.txt')
    #step 2) Initialize paramter for model
    parameters=initializeParameter(filedata)
    #step 3) visualizeData Data using graph
    visualizeData(filedata,parameters)
    #step 4) Calculate initial Error cost
    x=filedata[:, 0]#0 for get first column
    y=filedata[:, 1]#1 for get second column
    Q0=parameters["Q0"]
    Q1=parameters["Q1"]
    datasize=y.size#Number of rows in data

    initialcost=cost(x,y,Q0,Q1,datasize)  #initialy cost calculation
    print("Initialy ERROR Cost For Q0 %d and %d Q1 is %d ==> "%(Q0,Q1,initialcost))
    #step 5) Apply Gradient descent for n interation
    Finalq0q1=gradient_descent(x,y,Q0,Q1,parameters)
    finalcost=cost(x,y,Finalq0q1[0],Finalq0q1[1],datasize)
    print("Initialy ERROR Cost For Q0 %d and %d Q1 is %d ==> "%(Finalq0q1[0],Finalq0q1[1],finalcost))
    #Note: now you have Q0, Q1 value,Congrets your model is trained, now go for testing 
    #So Y=Q0+Q1X ,Put some x value in euqation now

except:
    print "Exception occured:"
    traceback.print_exc()



    


