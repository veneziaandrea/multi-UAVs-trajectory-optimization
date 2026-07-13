import pandas as pd

'''
Produces latex code in terminal to be copied and pasted.
Requires Jinja2 package to be installed with pip.
'''

# 1. Load the data 
df_all = pd.read_csv('logs/position_noise_30cm.csv')
df_all.columns = df_all.columns.str.strip()
df_all['Final_State'] = df_all['Final_State'].str.strip() 

# 2. Compute Success Rate using ALL logs
df_all['Success_Rate_pct'] = (df_all['Final_State'] == 'Success').astype(float) * 100.0
success_means = df_all.groupby('Algorithm')['Success_Rate_pct'].mean()
success_stds = df_all.groupby('Algorithm')['Success_Rate_pct'].std()

# 3. Filter for ONLY successful missions for the remaining metrics
df_success = df_all[df_all['Final_State'] == 'Success'].copy()

# 4. Define the exact numeric columns (excluding Success Rate)
metrics = [
    'Speed_m_s', 'Jerk_m2_s5', 'Miss_Distance_m', 
    'Energy_Joules', 'Flight_Time_s', 'Collisions', 
    'Coverage_pct', 'Solve_Time'
]

# 5. Calculate means and standard deviations on SUCCESSFUL logs only
means = df_success.groupby('Algorithm')[metrics].mean().T
stds = df_success.groupby('Algorithm')[metrics].std().T

# Append the success rate (calculated from df_all) to the bottom of our results
means.loc['Success_Rate_pct'] = success_means
stds.loc['Success_Rate_pct'] = success_stds

# Enforce column order
means = means[['Normal', 'Early']]
stds = stds[['Normal', 'Early']]

# 6. Combine into a single string format: "Mean \pm Std"
summary = pd.DataFrame(index=means.index)
for col in means.columns:
    summary[col] = means[col].map('{:.3f}'.format) + ' $\\pm$ ' + stds[col].map('{:.3f}'.format)

# 7. Rename the row index to look professional in LaTeX
summary.index = [
    'Speed (m/s)', 
    'Jerk ($m^2/s^5$)', 
    'Miss Distance (m)', 
    'Energy (J)', 
    'Flight Time (s)', 
    'Collisions', 
    'Coverage (\\%)', 
    'Solve Time (s)',
    'Success Rate (\\%)'
]

# 8. Generate LaTeX code
latex_code = summary.to_latex(
    column_format='lcc', 
    caption='Comparison of Normal vs. Early Switching Algorithms (Successful Missions Only)',
    label='tab:early_vs_normal',
    escape=False 
)

# Convert to standard LaTeX table rules (bypassing Pandas version issues)
latex_code = latex_code.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline')
latex_code = latex_code.replace('\\hline', '\\toprule', 1) 
latex_code = latex_code[::-1].replace('enilh\\', 'elurmottob\\', 1)[::-1] 
latex_code = latex_code.replace('\\hline', '\\midrule') 

print(latex_code)