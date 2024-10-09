# -*- coding: utf-8 -*-
"""
Meta Analysis Codes-created by Peiyan
"""

pip install pymeta # pymeta package in Python has the similar funtion of metafor in R
pip install metaplot # For creating forest plots and other meta-analysis visualizations.
pip install openpyxl
pip install xlsxwriter
pip install statsmodels


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from scipy.stats import chi2
from scipy.stats import kendalltau
from scipy.stats import norm
#from pymeta.core import MetaAnalysis


print("Current Working Directory:", os.getcwd())

path = "/Users/peiyan/Documents/GWU/24_Summer/Research_Meta/Meta_Analysis"
os.chdir(path)


input = pd.read_excel('Meta_2 copy.xlsx', sheet_name='2nd coding',engine='openpyxl')
input.columns

df = input[['Hedgesg', 'Standard deviation','StandardError','Blooms_Taxonomy_codes','Teaching/Leaning_codes']]
df['std'] = df['Standard deviation'].str.replace(r'\s*\(.*?\)\s*', '', regex=True)
df['std'] = df['std'].astype(float)
df['var'] = df['std']**2

df = df[['Hedgesg', 'std','var','StandardError','Blooms_Taxonomy_codes','Teaching/Leaning_codes']]
df = df.dropna()
df= df.drop(12)#code to drop the extremem large effect size


# Fixed effect model without moderators
fixed_effect_model = sm.WLS(df['Hedgesg'], sm.add_constant(df.index), weights=1/df['std']**2)
fixed_result = fixed_effect_model.fit()
print(fixed_result.summary())


# Random effect model without moderators
exog = sm.add_constant(df.index)
endog = df['Hedgesg']

random_effect_model = sm.MixedLM(endog, exog, groups=df.index)
random_result = random_effect_model.fit()
print(random_result.summary())

# Calculate AIC and BIC manually
log_likelihood = random_result.llf
num_params = random_result.df_modelwc
aic = -2 * log_likelihood + 2 * num_params
bic = -2 * log_likelihood + num_params * np.log(len(endog))

# Print AIC and BIC
print(f"AIC: {aic}")
print(f"BIC: {bic}")





# Random effect model with bloom's as one moderator
exog_with_mod1 = sm.add_constant(df[['Blooms_Taxonomy_codes']])
random_effect_model_with_mod1 = sm.MixedLM(endog, exog_with_mod1, groups=df.index)
random_result_with_mod1 = random_effect_model_with_mod1.fit()
print(random_result_with_mod1.summary())


# Calculate AIC and BIC manually
log_likelihood = random_result_with_mod1.llf
num_params = random_result_with_mod1.df_modelwc
aic = -2 * log_likelihood + 2 * num_params
bic = -2 * log_likelihood + num_params * np.log(len(endog))

# Print AIC and BIC
print(f"AIC: {aic}")
print(f"BIC: {bic}")




# Random effect model with teaching/learning as one moderator
exog_with_mod2 = sm.add_constant(df[['Teaching/Leaning_codes']])
random_effect_model_with_mod2 = sm.MixedLM(endog, exog_with_mod2, groups=df.index)
random_result_with_mod2 = random_effect_model_with_mod2.fit()
print(random_result_with_mod2.summary())

# Calculate AIC and BIC manually
log_likelihood = random_result_with_mod2.llf
num_params = random_result_with_mod2.df_modelwc
aic = -2 * log_likelihood + 2 * num_params
bic = -2 * log_likelihood + num_params * np.log(len(endog))

# Print AIC and BIC
print(f"AIC: {aic}")
print(f"BIC: {bic}")







# Random effect model with moderators
exog_with_mod = sm.add_constant(df[['Blooms_Taxonomy_codes', 'Teaching/Leaning_codes']])
random_effect_model_with_mod = sm.MixedLM(endog, exog_with_mod, groups=df.index)
random_result_with_mod = random_effect_model_with_mod.fit()
print(random_result_with_mod.summary())

# Calculate AIC and BIC manually
log_likelihood = random_result_with_mod.llf
num_params = random_result_with_mod.df_modelwc
aic = -2 * log_likelihood + 2 * num_params
bic = -2 * log_likelihood + num_params * np.log(len(endog))

# Print AIC and BIC
print(f"AIC: {aic}")
print(f"BIC: {bic}")





#q test calculation

# Step 1: Calculate the variance and the weights (1 / variance)
df['weight'] = 1 / df['var']
# Step 2: Calculate the weighted mean effect size
weighted_mean = np.sum(df['Hedgesg'] * df['weight']) / np.sum(df['weight'])

# Step 3: Calculate the Q statistic
df['squared_diff'] = df['weight'] * (df['Hedgesg'] - weighted_mean) ** 2
Q = np.sum(df['squared_diff'])

# Step 4: Degrees of freedom (k - 1)
k = len(df)  # number of studies
df_q = k - 1


# Step 5: P-value from chi-square distribution
p_value = chi2.sf(Q, df_q)


# Output the results
print(f'Weighted Mean Effect Size: {weighted_mean}')
print(f'Q statistic: {Q}')
print(f'Degrees of freedom: {df_q}')
print(f'P-value: {p_value}')





# Funnel plot
# Create the funnel plot
plt.figure(figsize=(8, 6))

# Scatter plot of the effect sizes against the standard deviation (or precision)
plt.scatter(df['Hedgesg'], df['StandardError'], alpha=0.5)

# Add the vertical line at effect size = 0 (null effect)
plt.axvline(x=0, linestyle='--', color='gray')

# Add the 95% confidence interval funnel boundaries
# Calculate precision (1 / standard deviation)
precision = 1 / df['StandardError']

# Confidence intervals based on standard normal distribution
z_value = 1.96  # 95% confidence interval

# Create upper and lower bounds for the 95% CI
lower_bound = z_value * (1 / precision)
upper_bound = -z_value * (1 / precision)

# Plot the confidence intervals as lines forming the funnel
plt.plot(lower_bound, df['StandardError'], linestyle='--', color='red', label='95% CI')
plt.plot(upper_bound, df['StandardError'], linestyle='--', color='red')

# Labels and title
plt.title('Funnel Plot with 95% Confidence Interval')
plt.xlabel('Effect Size')
plt.ylabel('Precision (Standard Error)')
plt.legend()

# Display the plot
plt.show()



# Calculate Kendall's tau to check for publication bias
tau, p_value = kendalltau(df['Hedgesg'], df['std'])

# Output Kendall's tau and p-value
print(f"Kendall's Tau: {tau}")
print(f"P-value: {p_value}")


#calculate FailSafe N
# Step 1: Calculate the weighted mean effect size
weighted_mean = np.sum(df['Hedgesg'] * df['weight']) / np.sum(df['weight'])

# Step 2: Calculate the standard error of the weighted mean
weighted_se = np.sqrt(1 / np.sum(df['weight']))

# Step 3: Calculate the Z-value for the overall effect size
z_value = weighted_mean / weighted_se

# Step 4: Set the critical Z-value for a 95% confidence interval (Z_alpha/2 = 1.96)
z_alpha = 1.96

# Step 5: Calculate Fail-safe N
n_studies = len(df)  # Number of studies
failsafe_n = ((n_studies * z_value**2) - (z_alpha**2)) / (z_alpha**2)

# Output the Fail-safe N and other values
print(f'Weighted Mean Effect Size: {weighted_mean}')
print(f'Standard Error of the Weighted Mean: {weighted_se}')
print(f'Z-value: {z_value}')
print(f'Fail-safe N: {failsafe_n}')


'''
Weighted Mean Effect Size: 1.4683820818849656
Standard Error of the Weighted Mean: 0.13699267728744238
Z-value: 10.718690304912865
Fail-safe N: 507.4171885398693
'''




# Forest plot
plt.figure(figsize=(8, 4))
effect_sizes = df['Hedgesg']
std_errors = df['StandardError']
study_names = [f'Study {i}' for i in [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18]]

#study_names = [f'Study {i+1}' for i in range(len(df))]

plt.errorbar(effect_sizes, range(len(df)), xerr=std_errors, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
plt.yticks(range(len(df)), study_names)
plt.axvline(x=0, linestyle='--', color='gray')
plt.title('Forest Plot')
plt.xlabel('Effect Size')
plt.ylabel('Study')
plt.show()
















