import numpy as np
import matplotlib.pyplot as plt

####
# Functions for 3_1
####

def initialize(dataset,numOutputs=1):
    """
    Initialize a training run for any of the Hebb-style rules in lab3_1
    on either dataset D1 or D2
    This means we: 
         Set up the figure for an animated plot and
         Return initial weight vectors
    
    Parameters
    ----------
    dataset      : numpy array, either D1 or D2
    numOutputs   : int, num(ber of )Output( Neuron)s
    
    Returns
    -------
    figure               : matplotlib figure object, base for animation
    plottedWeightVectors : matplotlib line object, will update data during animation
    weights              : numpy array, weight matrix of linear transformation of input data
    
    also, draws the dataset with plottedWeightVectors on top
    """
    
    numInputs,numDatapoints = dataset.shape
    
    # plot data
    figure = plt.figure(figsize=(8,8));
    plt.plot(dataset[0,:],dataset[1,:],'.',alpha=0.2,color='gray',markersize=10)
    
    # initialize weights
    weights = np.random.randn(numInputs,numOutputs) 
    
    # plot weight vector
    
    plottedWeightVectors = [0]*numOutputs
    for i in range(numOutputs):
        plottedWeightVectors[i], = plt.plot([0,weights[0,i]],[0,weights[1,i]],
                                    'r',linewidth=4,alpha=0.8);
    plt.draw()
    
    return figure,plottedWeightVectors,weights

def doLearn(learnFunk,dataset,figure,plottedWeightVectors,weights,learningRate,numTrials):
    """
    updates weights according to learnFunk on dataset numTrials times,
    updating plottedWeightVectors in figure each time
    
    Parameters
    ----------
    learnFunk            : function, a Hebbian-type learning function
    dataset              : numpy array, either D1 or D2
    figure               : matplotlib figure, will draw into this canvas
    plottedWeightVectors : matplotlib line object, will update data during animation
    weights              : numpy array, weight matrix of linear transformation of input data
    learningRate         : float, scaling factor for gradients
    
    Returns
    -------
    weights
    
    and draws into provided figure
    """
    
    for trial in range(numTrials):
        animFrame(learnFunk,dataset,plottedWeightVectors,weights,learningRate)
        figure.canvas.draw()
    
    return weights
    

def animFrame(learnFunk,dataset,plottedWeightVectors,weights,learningRate):
    """
    this function will be called each time you run a trial,
    and it will produce another frame of animation each time
    
    Parameters
    ----------
    learnFunk            : function, a Hebbian-type learning function
    dataset              : numpy array, either D1 or D2
    plottedWeightVectors : matplotlib line object, will update data during animation
    weights              : numpy array, weight matrix of linear transformation of input data
    learningRate         : float, scaling factor for gradients
    
    Returns
    -------
    None
    
    but edits figure provided to doLearn
    """

    weights = learnFunk(dataset,weights,learningRate)
    
    for idx,plottedWeightVector in enumerate(plottedWeightVectors):
        plottedWeightVector.set_data([0,weights[0,idx]], [0,weights[1,idx]]) # replot weight vector
    return

    #plottedWeightVector[0].set_data([0,weights[0]], [0,weights[1]]) # replot weight vector
    
    return

def initializeWTA(dataset,goodWeights=False,numOutputs=1):
    """
    Initialize a training run for a winner-take-all network
    on either dataset D1 or D2
    This means we: 
         Set up the figure for an animated plot and
         Return initial weight vectors
    
    Parameters
    ----------
    dataset      : numpy array, either D1 or D2
    goodWeights  : bool, if True, use non-random weights that work
    numOutputs   : int, num(ber of )Output( Neuron)s
    
    Returns
    -------
    figure               : matplotlib figure object, base for animation
    plottedWeightVectors : matplotlib line object, will update data during animation
    weights              : numpy array, weight matrix of linear transformation of input data
    
    also, draws the dataset with plottedWeightVectors on top
    """
    
    colorList = ['red','yellow','black','blue','indigo','brown','gray',
             'skyblue','tomato','mediumspringgreen',
            'plum','darkcyan','orange','darkolivegreen','hotpink']
    
    numInputs,numDatapoints = dataset.shape
    
    # plot data
    figure = plt.figure(figsize=(8,8));
    plt.plot(dataset[0,:],dataset[1,:],'.',alpha=0.2,color='gray',markersize=10)
    
    # initialize weights
    if goodWeights:
        weights = np.asarray([[1,0],[0,1],[0,-1],[-1,0]],dtype=np.float32).T 
    else:
        weights = np.random.randn(numInputs,numOutputs) 
    
    # plot weight vector
    
    plottedWeightVectors = [0]*numOutputs
    for i in range(numOutputs):
        plottedWeightVectors[i], = plt.plot([0,weights[0,i]],[0,weights[1,i]],
                                    'r',linewidth=4,alpha=0.8,
                                           color = colorList[i]);
    plt.draw()
    
    return figure,plottedWeightVectors,weights

####
# Functions for 3_2
####

# A Whole Bunch of Convenience Functions for Cleaning Up Plots
def removeAxes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def removeFrames(ax,sides=['top','right']):
    for side in sides:
        ax.spines[side].set_visible(False)

def removeTicks(ax,axes):
    if 'x' in axes:
        ax.tick_params(axis='x',
                        which='both',
                        top='off',
                        labeltop='off',
                        bottom='off',
                        labelbottom='off')
    if 'y' in axes:
        ax.tick_params(axis='y',
                        which='both',
                        left='off',
                        labelleft='off',
                        right='off',
                        labelright='off')

def addAxis(ax,axis='horizontal'):
    if axis == 'horizontal':
        xmin,xmax = ax.get_xlim()
        ax.hlines(0,xmin,xmax)
    elif axis == 'vertical':
        ymin,ymax = ax.get_ylim()
        ax.vlines(0,ymin,ymax)

def cleanPlot(ax):
    removeFrames(plt.gca(),['top','right','bottom']);
    removeTicks(plt.gca(),['x','y']);

def setLims(ax,xBounds,yBounds):
    ax.set_xlim(xBounds); ax.set_ylim(yBounds);

###    

def faceReshape(faceColumn):
    """
    Reshapes a column-vector yaleface into a matrix yaleface
    suitable for viewing as an image w facePlot
    
    Parameters
    ----------
    faceColumn : numpy array, column vector from yalefaces
    
    Returns
    -------
    faceImage  : numpy array, reshaped faceColumn
    """
    faceImage = np.reshape(faceColumn,(61,-1),order='C')
    return faceImage 

def facePlot(faceImage):
    """
    Plots a faceImage made by faceReshape
    in an axis with no ticks or labels
    
    Parameters
    ----------
    faceImage   : numpy array, reshaped faceColumn from faceReshape
    
    Returns
    -------
    imageHandle : matplotlib Image object, for use with animation routines
    """
    
    imageHandle = plt.imshow(faceImage,cmap='Greys_r'); 
    cleanPlot(plt.gca()) #remove ticks etc.
    return imageHandle

def faceInitialize(dataset,numOutputs=1):
    """
    Initializes a run for Sanger's rule on the yalefaces dataset
    
    Parameters
    ----------
    dataset      : numpy array, yalefaces
    numOutputs   : int, number of PCs to learn with Sanger's Rule
    
    Returns
    -------
    dataset      : numpy array, mean-subtracted yalefaces
    weights      : numpy array, dataset.shape[0] by numOutputs
    
    """
    # initialize a run for Sanger's rule on the faces dataset 
    # takes in a dataset and optionally the number of outputs
    # (number of eigenfaces to find) and the learning rate
    
    numInputs,numDatapoints = dataset.shape
    
    #subtract average
    dataset = dataset - np.atleast_2d(np.mean(dataset,1)).T

    # initialize weights
    weights = np.random.randn(numInputs,numOutputs)
    
    return dataset,weights

def pickEigenvector(idx,weights):
    """
    Pulls idx'd column from weights and returns ±column
    """
    vector = weights[:,idx]
    target = -vector
    return vector, target

def eigenFrame(frameIdx,imageHandle,vector,target,numFrames,averageFace,empiricalNorm):
    """
    Animation routine that interpolates between vector and target over numFrames
    """
    #linear interpolation coefficients:
    a = empiricalNorm*(numFrames-frameIdx)/numFrames #goes from empiricalNorm to 0
    b = (empiricalNorm - a) # goes from 0 to empiricalNorm
    imageHandle.set_data(faceReshape((a*vector+b*target+averageFace))) #mix from initial to final
