#!/usr/bin/env python
# encoding: utf-8
"""
helpers.py

Created by Safak Ozkan on 2017-02-04.
Copyright (c) 2017 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import numpy as np
from scipy.stats import norm, chisquare, chi2, chi2_contingency


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
	
	
	
	
	
	
	
	
	
	
		
		
		