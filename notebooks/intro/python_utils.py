import scipy.io
import numpy as np
#import mat73
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import random
import matplotlib.colors as mcolors
import scipy.io
from mpl_toolkits import mplot3d
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import matplotlib as mpl
import matplotlib.colors as colors
import scipy.stats as stats
#import pandas as pd
import scipy
#from statannot import add_stat_annotation
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os


def extractSpike(spikeData):
    VAR=[]
    if len(spikeData)>=2:
        for k in range(len(spikeData)):
            VAR.append(np.array(list(spikeData[k])).T,)
            
    else: VAR.append(np.array(list(spikeData)).T,)
    return VAR


def recover_results(outputs):
    results = {}
    for key in outputs.keys(): # to extract the name of the layer, e.g., Exc, Inh, Thalamus, etc  
        print(key)
        # to get voltage and conductances
        for analogsignal in outputs[key].segments[0].analogsignals:
            print(analogsignal.name)
            results[key, analogsignal.name] = analogsignal

        # to get spikes
        VAR=outputs[key].segments[0].spiketrains
        results[key, 'spikes']=[]
        for k in range(len(VAR)):
            results[key, 'spikes'].append(np.array(list(VAR[k])).T,)
    return results