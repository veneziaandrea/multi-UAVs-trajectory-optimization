import pandas as pd

# 1. Load the data 
df = pd.read_csv('logs/position_noise_20cm.csv')
df.columns = df.columns.str.strip()

df['Final_State'] = df['Final_State'].str.strip() # Strip accidental spaces

# 2. Convert text states to a numerical success percentage (100 or 0)
df['Success_Rate_pct'] = (df['Final_State'] == 'Success').astype(float) * 100.0

# 2. Define the exact numeric columns
metrics = [
    'Speed_m_s', 'Jerk_m2_s5', 'Miss_Distance_m', 
    'Energy_Joules', 'Flight_Time_s', 'Collisions', 
    'Coverage_pct', 'Solve_Time', 'Success_Rate_pct'
]

# 3. Calculate means and standard deviations, then transpose
means = df.groupby('Algorithm')[metrics].mean().T
stds = df.groupby('Algorithm')[metrics].std().T

# Enforce column order
means = means[['Normal', 'Early']]
stds = stds[['Normal', 'Early']]

# 4. Combine into a single string format: "Mean \pm Std"
summary = pd.DataFrame(index=means.index)
for col in means.columns:
    summary[col] = means[col].map('{:.3f}'.format) + ' $\\pm$ ' + stds[col].map('{:.3f}'.format)

# 5. Rename the row index to look professional in LaTeX
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

# 6. Generate LaTeX code
# escape=False is required so Pandas doesn't break the $\pm$ symbol
latex_code = summary.to_latex(
    column_format='lcc', 
    caption='Comparison of Normal vs. Early Switching Algorithms (Mean $\\pm$ Std)',
    label='tab:early_vs_normal',
    escape=False
)

latex_code = latex_code.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline')
latex_code = latex_code.replace('\\hline', '\\toprule', 1) # First line to toprule
latex_code = latex_code[::-1].replace('enilh\\', 'elurmottob\\', 1)[::-1] # Last line to bottomrule
latex_code = latex_code.replace('\\hline', '\\midrule') # Middle lines to midrule

print(latex_code)