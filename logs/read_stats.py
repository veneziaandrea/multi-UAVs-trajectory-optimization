import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_offline_csv_comparison(csv_filepath):
    df = pd.read_csv(csv_filepath)
    df_success = df  
    
    algos = df_success['Algorithm'].unique()
    if len(algos) != 2: return
    name_a, name_b = algos[0], algos[1]
    
    global_df = df_success[['Map_Seed', 'Algorithm', 'Coverage_pct']].drop_duplicates()
    global_stats = global_df.groupby('Algorithm')['Coverage_pct'].mean().to_dict()
    
    # ---> ADD COLLISIONS TO GROUPBY <---
    drone_stats = df_success.groupby(['Algorithm', 'Drone_ID'])[['Speed_m_s', 'Jerk_m2_s5', 'Energy_Joules', 'Flight_Time_s', 'Collisions']].agg(['mean', 'std']).reset_index()
    
    drone_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in drone_stats.columns.values]
    
    data_a = drone_stats[drone_stats['Algorithm'] == name_a].sort_values('Drone_ID')
    data_b = drone_stats[drone_stats['Algorithm'] == name_b].sort_values('Drone_ID')
    drone_ids = data_a['Drone_ID'].tolist()
    
    x = np.arange(len(drone_ids))
    width = 0.35
    
    # ---> UPGRADED TO 5 SUBPLOTS (Taller Figure) <---
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 20))
    
    title_str = (f"Offline Trajectory Analysis: {name_a} vs {name_b} (Averaged across maps)\n"
                 f"Mean Coverage: {name_a} ({global_stats.get(name_a, 0):.2f}%) vs "
                 f"{name_b} ({global_stats.get(name_b, 0):.2f}%)")
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.97)
    
    color_a, color_b = '#2ca02c', '#1f77b4'
    
    # Subplot 1: Speed
    rects1_a = ax1.bar(x - width/2, data_a['Speed_m_s_mean'], width, yerr=data_a['Speed_m_s_std'], capsize=5, label=name_a, color=color_a, edgecolor='black')
    rects1_b = ax1.bar(x + width/2, data_b['Speed_m_s_mean'], width, yerr=data_b['Speed_m_s_std'], capsize=5, label=name_b, color=color_b, edgecolor='black')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Average Cornering Speed')
    ax1.set_xticks(x); ax1.set_xticklabels(drone_ids)
    ax1.legend(); ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Subplot 2: Jerk
    rects2_a = ax2.bar(x - width/2, data_a['Jerk_m2_s5_mean'], width, yerr=data_a['Jerk_m2_s5_std'], capsize=5, color=color_a, edgecolor='black')
    rects2_b = ax2.bar(x + width/2, data_b['Jerk_m2_s5_mean'], width, yerr=data_b['Jerk_m2_s5_std'], capsize=5, color=color_b, edgecolor='black')
    ax2.set_ylabel('Jerk ($m^2/s^5$)')
    ax2.set_title('Average Cumulative Jerk')
    ax2.set_xticks(x); ax2.set_xticklabels(drone_ids)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Subplot 3: Energy
    rects3_a = ax3.bar(x - width/2, data_a['Energy_Joules_mean'], width, yerr=data_a['Energy_Joules_std'], capsize=5, color=color_a, edgecolor='black')
    rects3_b = ax3.bar(x + width/2, data_b['Energy_Joules_mean'], width, yerr=data_b['Energy_Joules_std'], capsize=5, color=color_b, edgecolor='black')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Average Mechanical Energy Expended')
    ax3.set_xticks(x); ax3.set_xticklabels(drone_ids)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Subplot 4: Time
    rects4_a = ax4.bar(x - width/2, data_a['Flight_Time_s_mean'], width, yerr=data_a['Flight_Time_s_std'], capsize=5, color=color_a, edgecolor='black')
    rects4_b = ax4.bar(x + width/2, data_b['Flight_Time_s_mean'], width, yerr=data_b['Flight_Time_s_std'], capsize=5, color=color_b, edgecolor='black')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Average Flight Time per Drone')
    ax4.set_xticks(x); ax4.set_xticklabels(drone_ids)
    
    min_time = min(data_a['Flight_Time_s_mean'].min(), data_b['Flight_Time_s_mean'].min())
    max_time_a = (data_a['Flight_Time_s_mean'] + data_a['Flight_Time_s_std']).max()
    max_time_b = (data_b['Flight_Time_s_mean'] + data_b['Flight_Time_s_std']).max()
    max_time = max(max_time_a, max_time_b)
    padding = (max_time - min_time) * 0.5 if max_time != min_time else 5
    ax4.set_ylim(max(0, min_time - padding), max_time + padding)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Subplot 5: Collisions (THE NEW METRIC) ---
    rects5_a = ax5.bar(x - width/2, data_a['Collisions_mean'], width, yerr=data_a['Collisions_std'], capsize=5, color=color_a, edgecolor='black')
    rects5_b = ax5.bar(x + width/2, data_b['Collisions_mean'], width, yerr=data_b['Collisions_std'], capsize=5, color=color_b, edgecolor='black')
    ax5.set_ylabel('Events')
    ax5.set_title('Average Collision Events (Constraint Violations)')
    ax5.set_xticks(x); ax5.set_xticklabels(drone_ids)
    
    # Add a slight padding to the Y-axis to prevent flat bars if collisions are 0
    max_col = max((data_a['Collisions_mean'] + data_a['Collisions_std']).max(), (data_b['Collisions_mean'] + data_b['Collisions_std']).max())
    ax5.set_ylim(0, max_col + 1 if max_col > 0 else 1)
    ax5.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects, std_data, ax, format_str='{:.2f}'):
        for rect, std in zip(rects, std_data):
            height = rect.get_height()
            text_y_pos = height + (std if not np.isnan(std) else 0) 
            ax.annotate(format_str.format(height), xy=(rect.get_x() + rect.get_width() / 2, text_y_pos),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    autolabel(rects1_a, data_a['Speed_m_s_std'], ax1)
    autolabel(rects1_b, data_b['Speed_m_s_std'], ax1)
    autolabel(rects2_a, data_a['Jerk_m2_s5_std'], ax2, '{:.0f}')
    autolabel(rects2_b, data_b['Jerk_m2_s5_std'], ax2, '{:.0f}')
    autolabel(rects3_a, data_a['Energy_Joules_std'], ax3, '{:.0f}')
    autolabel(rects3_b, data_b['Energy_Joules_std'], ax3, '{:.0f}')
    autolabel(rects4_a, data_a['Flight_Time_s_std'], ax4, '{:.1f}')
    autolabel(rects4_b, data_b['Flight_Time_s_std'], ax4, '{:.1f}')
    autolabel(rects5_a, data_a['Collisions_std'], ax5, '{:.1f}')
    autolabel(rects5_b, data_b['Collisions_std'], ax5, '{:.1f}')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    filepath = "logs/switch_stats_40obs_distx2.csv"
    plot_offline_csv_comparison(filepath)