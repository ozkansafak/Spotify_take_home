#!/usr/bin/env python
# encoding: utf-8
"""
helpers.py

Created by Safak Ozkan on 2017-02-04.
Copyright (c) 2017 __MyCompanyName__. All rights reserved.
"""

from __future__ import division
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency
import matplotlib.pyplot as plt


def mu_sigma(n, N):
	mu = float(n) / N
	sigma = np.sqrt(mu * (1 - mu) / float(N))
	return mu, sigma

def a_b_test(n1, N1, n2, N2):
	# Ho: mu_A = mu_B
	# Let z = X_A - X_B
	# mean(z) = mu_A - mu_B
	# std(z) = sqrt(sigma_A**2 + sigma_B**2)
	
	mu_A, sigma_A = mu_sigma(n_A, N_A)
	mu_B, sigma_B = mu_sigma(n_B, N_B)
	(mu_A - mu_B) / np.sqrt(sigma_A**2 + sigma_B**2)
	
def two_sided_p_value(x, mu, sigma):
	if x >= mu:
		return 2 * (1 - norm.cdf(x,mu,sigma))
	else:
		return 2 * (norm.cdf(x,mu,sigma))
		
		
def my_chisquare_p_value(n_m, n_f):
	O = np.vstack((n_m, n_f))
	RV_sum = [np.sum(n_m), np.sum(n_f)]
	sum_total = np.sum(RV_sum)
	chi_stat = 0.0
	E = np.zeros((2,len(n_m)))
	for i, category in enumerate(zip(n_m, n_f)):
		for j in [0,1]: # j=0 => m, j=1 => f
			E[j,i] = np.sum(category) * float(RV_sum[j]) / float(sum_total)
			chi_stat += (O[j,i] - E[j,i]) ** 2 / E[j,i]
			
	p_value = 1 - chi2.cdf(chi_stat, O.shape[1] - 1)
	
	return {'chi_stat':chi_stat, 'p_value':p_value}
	
	
	
def plotter(xlabel=None, ylabel=None, title=None, xlim=None, ylim=None):
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches((15,5))
    plt.grid('on')
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if xlim: plt.xlim((0, xlim));
    if ylim: plt.ylim((0, ylim));
    if title: plt.title(title)
    return fig, ax
    

def num_of_listens(df, skip_cutoff=10*1e3):
    dict_track= df.groupby('track_id').groups
    grouped = df.groupby('track_id')
    s = grouped.apply(lambda x: sum(x['ms_played'] >= skip_cutoff))
    """ s is a pd.Series with index: 'track_ID' and 
     values: 'count of records that match the above condition'. 
     value == 0 means the track was never listened to for more than skip_cutoff secs"""

    s_arr = np.array(s)
    num_of_interest = s_arr[s_arr > 1].shape[0]
    # print 'Question: What percentage of tracks have been listened to for one or more times?'
    print 'Answer: {}%'.format(round(num_of_interest / len(s_arr) * 100))
    return s_arr

def autolabel(rects, ax, draw=False):
    """
    Attach a text label above each bar displaying its height
    """
    
    heights = []
    for rect in rects:
        heights.append(rect.get_height())

    percent = [float(h)/sum(heights)*100.0 for h in heights]
    
    e1, e2 = plt.ylim()
    plt.ylim((e1, e2*1.09))
    
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if draw:
            ax.text(rect.get_x() + rect.get_width()/2., 1.03*height, \
                '%d' % round(percent[i]) + '%', \
                ha='center', va='bottom')
        
    return percent
	
def sorted_ls(dirpath):
    mtime = lambda f: os.stat(os.path.join(dirpath, f)).st_mtime
    return list(sorted(os.listdir(dirpath), key=mtime))
	
	
def bar_plot_session_length(df_new, column_name='country', rotation=0, sort=True,\
                            mean_or_sum='mean', width=.9, pct_label=False):
    # this function first groups by the "column_name"
    #               then calculates "sum" and "mean" of sessions lengths wrt 'country' column
    #               then ax.bar plot it.
    
    df_sessions = df_new[df_new['session'] == 'start']
    grouped = df_sessions['session_length'].groupby(df_new[column_name])
    g_sum  = grouped.apply(lambda x: np.sum(x)/60) # convert to [min]
    g_mean = grouped.apply(lambda x: np.mean(x)/60)
    
    if sort:
        g_sum.sort_values(ascending=False, inplace=True)
        g_mean.sort_values(ascending=False, inplace=True)
    
    # g_names will be used for plt.xticks
    g_names = g_mean.index if mean_or_sum == 'mean' else g_sum.index
        
    y = g_mean if mean_or_sum == 'mean' else g_sum

    plt.figure().set_size_inches((15, 5))
    ax = plt.gca()
    
    x = np.array(range(len(y)))
    rects = ax.bar(x+.5-width/2, y, color='b', width=width, alpha=.4)
    
    if pct_label:
        percent1 = autolabel(rects, ax=ax, draw=True)

    plt.xticks(x+.5, g_names, rotation=rotation)
    plt.ylabel('[min]')
    plt.grid('on')
    plt.xlim([0, max(x)+1])
    tit_bit = 'average' if mean_or_sum == 'mean' else 'aggregate'
    plt.title(tit_bit + ' session lengths wrt to '+ column_name +' [min]');
		
		
		