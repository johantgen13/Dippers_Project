#!/usr/bin/env python
#
# This is a package that consists of all the code I have written for my 
# research summer 2024-2025. This project involved looking for "dippers"
# in archival ASAS-SN data.
#
# This is the first iteration of this compilation.
# Last Edited: 04/21/2025
# Author: Brayden JoHantgen

# Importing the necessary modules
from wise_light_curves.wise_light_curves import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import math
import scipy
import os
from tqdm import tqdm
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from astropy.io import ascii
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import constants as const
from astropy.timeseries import LombScargle as ls
# End of Importing

# This function makes plots nicer (was written by Dom)
def plotparams(ax, labelsize=15):
    '''
    Basic plot params

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :returns: modified matplotlib axes object
    '''

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax
# End of plotparams function

# This general function reads ASAS-SN light curves
def read_lightcurve(asas_sn_id, guide = 'known_dipper_lightcurves/', ftype = '.csv', 
					column_names = ["JD", "UT Date", "Camera", "FWHM", "Limit", "Mag", "Mag_err", "Flux", "Flux_err", "Filter"]):
    """
    Inputs: 
    	asas_sn_id: id of desired star 
    	guide: file path
    	ftype: file type
    	column_names: file column names

    Outputs:
    	dfv: The V-band data of the ASAS-SN light curve 
    	dfg: The g-band data of the ASAS-SN light curve 

    Description:
    	This function reads the data of the desired star by going to the corresponding file and copying the data of that file onto a
    	data frame. This data frame is then sorted into two data frames by comparing the value in the Photo filter column. If the Photo 
    	filter column data has a value of one, its row is sorted into the dataframe corresponding to the V-band. If the Photo filter column 
    	data has a value of zero, it gets sorted into the dataframe corresponding to the g-band.
    """
    
    fname = os.path.join(guide, str(asas_sn_id)+ftype)

    dfv = pd.DataFrame()
    dfg = pd.DataFrame()

    fdata = pd.read_csv(fname, header=None)
    fdata.columns = column_names

    dfv = fdata.loc[fdata["Filter"] == 'V'].reset_index(drop=True)
    dfg = fdata.loc[fdata["Filter"] == 'g'].reset_index(drop=True)

    return dfv, dfg
# End of the read_lightcurve function

def find_peak(df, prominence=0.17, distance=25, height=0.3, width=2):
	'''
	Inputs:
		df: dataframe of the data, requires columns of 'Mag' and 'JD'
		prominence: same parameter of scipy.signal.find_peaks()
		distance: same parameter of scipy.signal.find_peaks()
		height: same parameter of scipy.signal.find_peaks()
		width: same parameter of scipy.signal.find_peaks()

	Outputs:
		peak: a series of the peaks found
		meanmag: the average magnitude of the light curve
		length: the number of peaks found

	Description:
	'''
	df['Mag'] = [float(i) for i in df['Mag']]

	df['JD'] = [float(i) for i in df['JD']]

	mag = df['Mag']

	jd = df['JD']

	meanmag = sum(mag) / len(mag)

	df_mag_avg = [i - meanmag for i in mag]

	peaks = scipy.signal.find_peaks(df_mag_avg, prominence=prominence, distance=distance, height=height, width=width) 

	peak = peaks[0]

	prop = peaks[1]

	length = len(peak)

	peak = [int(i) for i in peak]

	peak = pd.Series(peak)

	return peak, meanmag, length	
# End of the find_peak















