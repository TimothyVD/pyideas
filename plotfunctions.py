# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:07:26 2013

@author: VHOEYS

pltofunctions to support visual inspection
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LinearLocator, NullLocator


def definedec(nummin,nummax):
    '''
    Help function to define the number of shown decimals
    '''
    diff = nummax - nummin
    predec = len(str(diff).split('.')[0])
    if predec > 1:
        dec = -(predec-1)
    else:
        if str(diff)[0] <> '0':
            dec = 1
        else:
            cnt = 1
            for char in str(diff).split('.')[1]:
                if char == '0':
                    cnt+=1
            dec = cnt
    return dec

def scatterplot_matrix(data1, plottext=None, data2 = False, limin = False, 
                             limax = False, diffstyle1 = None, 
                             diffstyle2 = None, plothist = False,
                             mstyles=['o','v','^','<','>','1\
                             ','2','3','4','s','x','+',',','_','|'], 
                             *args, **kwargs):
    """
    Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots.
    
    Still some work: *only half of it showing + hide subplots upper half
        
    Parameters
    -----------
    data1: ndarray
        numvars rows and numdata columns datapoints to compare,
        when only this dataset is given, the dat is plotted twice in the 
        graph
    data2: ndarray
        optional second dataset to put in the upper-part, whereas the 
        first dataset is putted in the lower part
    plottext: None | list
        list of strings woth the text to put for the variables, when no 
        histograms are needed
    limin: False | list 
        List of user defined minimal values for the different
        variables. When False, the min/max values are calculated
    limax: False | list 
        List of user defined maximal values for the different
        variables. When False, the min/max values are calculated 
    diffstyle1: None |list
        when every variable contains sub-groups, the diffstyle list gives 
        the number of elements to group the elements, different groups are
        given different color/style caracteristics automatically
    diffstyle2: None |list
        analgue to diffstyle1
    mstyles: list
        list of user defined symbols to use for different groups
    plothist: bool
        histogram is plotted in the middle of the data1 when True
    *args, **kwargs: arg
        arguments passed to the scatter method 
    
    Returns
    ---------
    fig: matplotlib.figure.Figure object
        figure containing the output
    axes: array of matplotlib.axes.AxesSubplot object
        enabled post-processing of the ax-elements
        
    Examples
    ---------
    >>> np.random.seed(1977)
    >>> numvars, numdata = 4, 1111
    >>> data1 = 5 * np.random.normal(loc=3.,scale=2.0,size=(numvars, numdata))
    >>> data2 = 50 * np.random.random((numvars, numdata))
    >>> fig,axes = scatterplot_matrix(data1, data2 = False,
            linestyle='none', marker='o', color='black', mfc='none', 
            diffstyle1=[555,556], plothist = True, plottext=['A','B','C','D'])
    >>> ax2add = axes[0,0]
    >>> ax2add.text(0.05,0.8,r'$SSE_{\alpha}$',transform = ax2add.transAxes,
                    fontsize=20)
    >>> 
    >>> fig,axes = scatterplot_matrix(data1, data2 = data2,
            linestyle='none', marker='o', color='black', mfc='none', 
            diffstyle1=False, plothist = False, plottext=['A','B','C','D'])   
    
    Notes
    ------
    typically used for comparing objective functions outputs, or parameter 
    distributions
    
    When using two datasets, only useful ticks when the datalimits
    are more or less the same, since otherwise the plot won't show both nicely
    """
       
    databoth = False
    if isinstance(data2, np.ndarray):
        databoth = True
    
    #TODO: control for inputs
    
    numvars, numdata = data1.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(40,40))
    fig.subplots_adjust(hspace=0.05, wspace=0.03)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
        #adjust the ticker lengths and position
        ax.tick_params(direction = 'out', pad=8, length = 5., 
                       color = 'black', which = 'major')
        ax.tick_params(length = 3., which = 'minor')

    #calc datalimits
    if not isinstance(limin, list) or not isinstance(limax,list):
        limin=[]
        limax=[]        
        for i in range(data1.shape[0]):
            if databoth == True:
                dec1 = definedec(np.min(data1[i]),np.max(data1[i]))
                dec2 = definedec(np.min(data2[i]),np.max(data2[i]))                
                limin1=np.around(np.min(data1[i]),decimals = dec1)
                limax1=np.around(np.max(data1[i]),decimals = dec1)
                limin2=np.around(np.min(data2[i]),decimals = dec2)
                limax2=np.around(np.max(data2[i]),decimals = dec2)
                print dec2
                limin.append(min(limin1,limin2))
                limax.append(max(limax1,limax2))
                               
                if np.abs(limin1 - limin2) > min(limin1,limin2):
                    print np.abs(limin1 - limin2), min(limin1,limin2),'min'
                    print 'potentially the datalimits of two datasets are \
                    too different for presenting results'
                if np.abs(limax1 - limax2) > min(limax1,limax2):
                    print np.abs(limax1 - limax2), min(limax1,limax2),'max'
                    print 'potentially the datalimits of two datasets are\
                    too different for acceptabel results'
            else:
                dec1 = definedec(np.min(data1[i]),np.max(data1[i]))
                limin.append(np.around(np.min(data1[i]),decimals = dec1))
                limax.append(np.around(np.max(data1[i]),decimals = dec1))
        print 'used limits are', limin,'and', limax
    else:
        print 'used limits are', limin,'and', limax          
    
    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
#        for x, y in [(i,j), (j,i)]:
        for x, y in [(j,i)]: #low
            if diffstyle1:
                cls=np.linspace(0.0,0.5,len(diffstyle1))              
                dfc=np.cumsum(np.array(diffstyle1))
                dfc=np.insert(dfc,0,0)        
                if dfc[-1]<>data1.shape[1]:
                    raise Exception('sum of element in each subarray is\
                    not matching the total data size')
                if len(diffstyle1)>15:
                    raise Exception('Not more than 15 markers provided')
                for ig in range(len(diffstyle1)):
                    axes[x,y].plot(data1[y][dfc[ig]:dfc[ig+1]], 
                                    data1[x][dfc[ig]:dfc[ig+1]], 
                                    marker=mstyles[ig], markersize = 6.,
                                    linestyle='none', markerfacecolor='none', markeredgewidth=0.7,#color=str(cls[ig]),
                                    markeredgecolor=str(cls[ig]))                   
                axes[x,y].set_ylim(limin[x],limax[x])
                axes[x,y].set_xlim(limin[y],limax[y])
                
            else:    
                axes[x,y].plot(data1[y], data1[x], *args, **kwargs)
                axes[x,y].set_ylim(limin[x],limax[x])
                axes[x,y].set_xlim(limin[y],limax[y])
                         
 
        if databoth == True: #plot data2
            for x, y in [(i,j)]: 
                if diffstyle2:
                    cls=np.linspace(0.0,0.5,len(diffstyle2))
                    dfc=np.cumsum(np.array(diffstyle2))
                    dfc=np.insert(dfc,0,0)        
                    if dfc[-1]<>data2.shape[1]:
                        raise Exception('sum of element in each subarray\
                        is not matching the total data size')
                    if len(diffstyle1)>15:
                        raise Exception('Not more than 15 markers provided')                    
                    for ig in range(len(diffstyle2)):
                        axes[x,y].plot(data2[y][dfc[ig]:dfc[ig+1]], 
                                        data2[x][dfc[ig]:dfc[ig+1]], 
                                        marker=mstyles[ig], markersize = 6,
                                        linestyle='none', markerfacecolor='none', markeredgewidth=0.7,
                                        markeredgecolor=str(cls[ig]))
                    axes[x,y].set_ylim(limin[x],limax[x])
                    axes[x,y].set_xlim(limin[y],limax[y])                    
                else:             
                    axes[x,y].plot(data2[y], data2[x], *args, **kwargs)  
                    axes[x,y].set_ylim(limin[x],limax[x])
                    axes[x,y].set_xlim(limin[y],limax[y]) 
                    
        else: #plot the data1 again
            for x, y in [(i,j)]:
                if diffstyle1:
                    cls=np.linspace(0.0,0.5,len(diffstyle1))
                    dfc=np.cumsum(np.array(diffstyle1))
                    dfc=np.insert(dfc,0,0)        
                    if dfc[-1]<>data1.shape[1]:
                        raise Exception('sum of element in each subarray\
                        is not matching the total data size')
                    if len(diffstyle1)>15:
                        raise Exception('Not more than 15 markers provided')                    
                    for ig in range(len(diffstyle1)):
                        axes[x,y].plot(data1[y][dfc[ig]:dfc[ig+1]], 
                                        data1[x][dfc[ig]:dfc[ig+1]], 
                                        marker=mstyles[ig], markersize = 6,
                                        linestyle='none', markerfacecolor='none', markeredgewidth=0.7,
                                        markeredgecolor=str(cls[ig]))
                    axes[x,y].set_ylim(limin[x],limax[x])
                    axes[x,y].set_xlim(limin[y],limax[y])                    
                else:             
                    axes[x,y].plot(data1[y], data1[x], *args, **kwargs)  
                    axes[x,y].set_ylim(limin[x],limax[x])
                    axes[x,y].set_xlim(limin[y],limax[y]) 

    
    #PLOT histograms  and variable names  
    #    for i, label in enumerate(plottext):   
    for i in range(numvars):
        if not plothist and plottext:
            label = plottext[i]
            axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center')
        else: #plot histogram in center
            if diffstyle1:
                dfc=np.cumsum(np.array(diffstyle1))
                dfc=np.insert(dfc,0,0)        
                if dfc[-1]<>data1.shape[1]:
                    raise Exception('sum of element in each subarray is\
                    not matching the total data size')
                cls=np.linspace(0.0,0.5,len(diffstyle1))
                
                for ig in range(len(diffstyle1)):
                    axes[i,i].hist(data1[i][dfc[ig]:dfc[ig+1]], 
                                    facecolor = 'none', bins=20, 
                                    edgecolor=str(cls[ig]), linewidth = 1.5)
                axes[i,i].set_xlim(limin[i],limax[i])      
                print limin[i],limax[i]
            else:
                axes[i,i].hist(data1[i],bins=20,color='k')
                axes[i,i].set_xlim(limin[i],limax[i])
                print limin[i],limax[i]
                
    if plothist:
        print 'plottext is not added'

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)
             
        majorLocator = LinearLocator(3)
        axes[j,i].xaxis.set_major_locator(majorLocator)
        axes[i,j].yaxis.set_major_locator(majorLocator)

        minorLocator  = LinearLocator(11)
        axes[j,i].xaxis.set_minor_locator(minorLocator)
        axes[i,j].yaxis.set_minor_locator(minorLocator)        
    
    #When uneven, some changes needed to properly put the ticks and tickslabels
    #since the ticks next to the histogram need to take the others y-scale
    #solved by adding a twinx taking over the last limits

    if not numvars%2==0:# and plothist==False:  
        if plothist == False:
            #create dummy info when no histogram is added
            axes[numvars-1,numvars-1].set_xlim(limin[numvars-1], 
                                        limax[numvars-1])
            axes[numvars-1,numvars-1].set_ylim(limin[numvars-1], 
                                        limax[numvars-1])
            
        axextra = axes[numvars-1,numvars-1].twinx()
        axextra.set_ylim(limin[numvars-1],limax[numvars-1])
        axextra.yaxis.set_minor_locator(minorLocator)
        axextra.yaxis.set_major_locator(majorLocator)
        axes[numvars-1,numvars-1].yaxis.set_ticks([])
        axes[numvars-1,numvars-1].yaxis.set_minor_locator(NullLocator())    
        
        axes[numvars-1,numvars-1].xaxis.set_major_locator(majorLocator)
        axes[numvars-1,numvars-1].xaxis.set_minor_locator(minorLocator)                
    return fig, axes   