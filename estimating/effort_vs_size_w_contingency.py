import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

adjusted_worst_offset = 0.3
adjusted_best_offset = -0.1

# Load the CSV file
df = pd.read_csv('_data\data.csv')

# Extract function points and effort days columns
function_points = df['Size'].values
effort_days = df['Effort'].values

# Function to calculate best and worst case effort scenarios
def calculate_scenarios(fp, effort):
    min_effort_per_fp = np.min(effort / fp)
    max_effort_per_fp = np.max(effort / fp)
    
    best_case = fp * min_effort_per_fp
    worst_case = fp * max_effort_per_fp
    
    return best_case, worst_case

# Generate best and worst-case scenarios
best_case_effort, worst_case_effort = calculate_scenarios(function_points, effort_days)

# Create the scatter plot
fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)

starting_x = min(function_points) - 2

# Manually set axis limits to avoid starting from 0
ax.set_xlim(left=starting_x, right=max(function_points) + 2)  # Add some padding if needed
ax.set_ylim(bottom=min(effort_days) - 2, top=max(effort_days) + 2)

scatter = ax.scatter(function_points, effort_days, color='blue', label='Actual Effort')

# Add labels, title, and grid
ax.set_xlabel('Function Points')
ax.set_ylabel('Effort (Days)')
ax.set_title('Historical Size vs Effort')
ax.grid(False)

ax2.set_xlabel('Project Budget (Days)')
ax2.set_ylabel('Effort (Days)')
ax2.set_title('Total Project Budget')
ax2.grid(True)

# Add crosshair lines (initially not visible)
hline = ax.axhline(y=0, color='gray', linestyle='--', visible=False)
vline = ax.axvline(x=0, color='gray', linestyle='--', visible=False)
annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# Add crosshair lines (initially not visible)
hline2 = ax2.axhline(y=0, color='gray', linestyle='--', visible=False)
vline2 = ax2.axvline(x=0, color='gray', linestyle='--', visible=False)
annot2 = ax2.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot2.set_visible(False)

# === Add Linear Trendline ===
# Fit a linear trendline (y = mx + b)
slope, intercept = np.polyfit(function_points, effort_days, 1)

# Generate y-values for the trendline based on the slope and intercept
trendline_y = slope * function_points + intercept

# Plot the trendline
ax.plot(function_points, trendline_y, color='purple', linestyle='-')

# Create placeholders for the horizontal lines (initially not visible)
hline_best, = ax.plot([], [], color='green', linestyle='-', visible=False)
htext_best = ax.text(0, 0, 'Best Case Effort', color='green', fontsize=8, ha='left', visible=False)
hline_best_adj, = ax.plot([], [], color='green', linestyle='-', visible=False)
htext_best_adj = ax.text(0, 0, 'Best Case Effort (adj)', color='green', fontsize=8, ha='left', visible=False)
hline_worst, = ax.plot([], [], color='red', linestyle='-', visible=False)
htext_worst = ax.text(0, 0, 'Worst Case Effort', color='red', fontsize=8, ha='left', visible=False)
hline_worst_adj, = ax.plot([], [], color='red', linestyle='-', visible=False)
htext_worst_adj = ax.text(0, 0, 'Worst Case Effort (adj)', color='red', fontsize=8, ha='left', visible=False)
hline_likely, = ax.plot([], [], color='blue', linestyle='-', visible=False)
htext_likely = ax.text(0, 0, 'Likely Effort', color='blue', fontsize=8, ha='left', visible=False)
vline_c, = ax.plot([], [], color='gray', linestyle='-', visible=False)
vtext_c = ax.text(0, 0, '', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', linewidth=1), fontsize=8, ha='center', visible=False)
vline_c_adj_b, = ax.plot([], [], color='gray', linestyle='-', visible=False)
vtext_c_adj_b = ax.text(0, 0, '', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', linewidth=1), fontsize=8, ha='center', visible=False)
vline_c_adj_w, = ax.plot([], [], color='gray', linestyle='-', visible=False)
vtext_c_adj_w = ax.text(0, 0, '', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', linewidth=1), fontsize=8, ha='center', visible=False)

cline, = ax2.plot([], [], color='red', linestyle='-', marker='.', label='Contingency Value', visible=False)
bline, = ax2.plot([], [], color='blue', linestyle='-', marker='.', label='Effort Value', visible=False)
tline, = ax2.plot([], [], color='green', linestyle='-', marker='o', label='Total Effort', visible=False)

hfund, = ax2.plot([], [], color='black', linestyle='-', visible=False)
vfund, = ax2.plot([], [], color='black', linestyle='-', visible=False)
tfund = ax2.text(0, 0, '', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', linewidth=1), fontsize=8, ha='center', visible=False)


def find_worst_effort(function_points, effort_days, specified_function_point, likely_effort, direction='L'):
    # Filter indices where function_points are less than the specified value
    if direction == 'L':
        valid_indices = np.where(function_points <= specified_function_point)[0]
    else:
        valid_indices = np.where(function_points >= specified_function_point)[0]
   
    # Get the corresponding effort_days values
    valid_effort_days = effort_days[valid_indices]
    valid_function_points = function_points[valid_indices]
    
    # Get the indexes of the effort_days which are greater than or equal to likely_effort
    valid_indices = np.where(valid_effort_days >= likely_effort)[0]
    valid_effort_days = valid_effort_days[valid_indices]
    valid_function_points = valid_function_points[valid_indices]
    
    if len(valid_indices) == 0:
        nearest_effort_day = 0
        nearest_function_point = specified_function_point
    else:
        # Sort the valid function points and get the two nearest values
        sorted_indices = np.argsort(valid_function_points)
        sorted_effort_days = valid_effort_days[sorted_indices]
        sorted_function_points = valid_function_points[sorted_indices]
      
        if len(sorted_effort_days) == 1:       
            nearest_effort_day = sorted_effort_days[-1]
            nearest_function_point = sorted_function_points[-1]
        elif sorted_effort_days[-1] >= sorted_effort_days[-2]:
            nearest_effort_day = sorted_effort_days[-1]
            nearest_function_point = sorted_function_points[-1]
        else:
            nearest_effort_day = sorted_effort_days[-2]
            nearest_function_point = sorted_function_points[-2]
    
    # Return the greater of the two nearest effort_days values
    return round(nearest_effort_day), round(nearest_function_point)

def find_best_effort(function_points, effort_days, specified_function_point, likely_effort, direction='L'):
    # Filter indices where function_points are less than the specified value
    if direction == 'L':
        valid_indices = np.where(function_points <= specified_function_point)[0]
    else:
        valid_indices = np.where(function_points >= specified_function_point)[0]
   
    # Get the corresponding effort_days values
    valid_effort_days = effort_days[valid_indices]
    valid_function_points = function_points[valid_indices]
    
    # Get the indexes of the effort_days which are greater than or equal to likely_effort
    valid_indices = np.where(valid_effort_days <= likely_effort)[0]
    valid_effort_days = valid_effort_days[valid_indices]
    valid_function_points = valid_function_points[valid_indices]
    
    if len(valid_indices) == 0:
        nearest_effort_day = 0
        nearest_function_point = specified_function_point
    else:
        # Sort the valid function points and get the two nearest values
        sorted_indices = np.argsort(valid_function_points)
        sorted_effort_days = valid_effort_days[sorted_indices]
        sorted_function_points = valid_function_points[sorted_indices]
      
        if len(sorted_effort_days) == 1:       
            nearest_effort_day = sorted_effort_days[-1]
            nearest_function_point = sorted_function_points[-1]
        elif sorted_effort_days[-1] <= sorted_effort_days[-2]:
            nearest_effort_day = sorted_effort_days[-1]
            nearest_function_point = sorted_function_points[-1]
        else:
            nearest_effort_day = sorted_effort_days[-2]
            nearest_function_point = sorted_function_points[-2]
    
    # Return the greater of the two nearest effort_days values
    return round(nearest_effort_day), round(nearest_function_point)

# Function to find the y value for a given x value on a line
def find_y_value(x1, y1, x2, y2, x_given):
    # Step 1: Calculate the slope (m)
    m = (y2 - y1) / (x2 - x1)
    
    # Step 2: Calculate the y-intercept (b)
    b = y1 - m * x1
    
    # Step 3: Calculate the y value for the given x value
    y_given = m * x_given + b
    
    return round(y_given)

# Function to handle click events
def on_click(event):
    # Check if click occurred inside the axes
    if event.inaxes == ax:
        # Get the clicked x-coordinate (FP value)
        clicked_x = round(event.xdata)
        
        # Calculate the most likely scenario (value on the trendline)
        likely_effort = round(slope * clicked_x + intercept)
        
        # Determine the worst-case scenario based on the clicked x-coordinate                
        wc_ef_left, wc_fp_left = find_worst_effort(function_points, effort_days, clicked_x, likely_effort)
        wc_ef_right, wc_fp_right = find_worst_effort(function_points, effort_days, clicked_x, likely_effort, direction='R')
        if wc_ef_left == 0:
            worst_case_effort = wc_ef_right
        elif wc_ef_right == 0:
            worst_case_effort = wc_ef_left
        else:
            worst_case_effort = find_y_value(wc_fp_left, wc_ef_left, wc_fp_right, wc_ef_right, clicked_x)
        
        # Determine the adjusted worst-case scenario based on the clicked x-coordinate  
        wc_adj_fp = round(clicked_x*(1+adjusted_worst_offset))             
        wc_ef_left, wc_fp_left = find_worst_effort(function_points, effort_days, wc_adj_fp, likely_effort)
        wc_ef_right, wc_fp_right = find_worst_effort(function_points, effort_days, wc_adj_fp, likely_effort, direction='R')
        if wc_ef_left == 0:
            worst_case_effort_adj = wc_ef_right
        elif wc_ef_right == 0:
            worst_case_effort_adj = wc_ef_left
        else:
            worst_case_effort_adj = find_y_value(wc_fp_left, wc_ef_left, wc_fp_right, wc_ef_right, wc_adj_fp)
        
        # Determine the best-case scenario based on the clicked x-coordinate                
        bc_ef_left, bc_fp_left = find_best_effort(function_points, effort_days, clicked_x, likely_effort)
        bc_ef_right, bc_fp_right = find_best_effort(function_points, effort_days, clicked_x, likely_effort, direction='R')
        if bc_ef_left == 0:
            best_case_effort = bc_ef_right
        elif bc_ef_right == 0:
            best_case_effort = bc_ef_left
        else:
            best_case_effort = find_y_value(bc_fp_left, bc_ef_left, bc_fp_right, bc_ef_right, clicked_x)
        
        # Determine the adjusted best-case scenario based on the clicked x-coordinate  
        bc_adj_fp = round(clicked_x*(1+adjusted_best_offset))              
        bc_ef_left, bc_fp_left = find_best_effort(function_points, effort_days, bc_adj_fp, likely_effort)
        bc_ef_right, bc_fp_right = find_best_effort(function_points, effort_days, bc_adj_fp, likely_effort, direction='R')
        if bc_ef_left == 0:
            best_case_effort_adj = bc_ef_right
        elif bc_ef_right == 0:
            best_case_effort_adj = bc_ef_left
        else:
            best_case_effort_adj = find_y_value(bc_fp_left, bc_ef_left, bc_fp_right, bc_ef_right, bc_adj_fp)
         
        print(f'Clicked FP: {clicked_x}, WC: {worst_case_effort}d, Adj WC {worst_case_effort_adj}d, BC: {best_case_effort}d, Adj BC: {best_case_effort_adj}d, Likely Effort: {likely_effort}d')

        # Update horizontal lines
        hline_best.set_data([starting_x, clicked_x], [best_case_effort, best_case_effort])
        hline_best.set_visible(True)
        htext_best.set_position((starting_x+1, best_case_effort + 2))
        htext_best.set_text(f'Best Case Effort: {best_case_effort}d')
        htext_best.set_visible(True)
        
        hline_best_adj.set_data([starting_x, bc_adj_fp], [best_case_effort_adj, best_case_effort_adj])
        hline_best_adj.set_visible(True)
        htext_best_adj.set_position((starting_x+1, best_case_effort_adj - 12))
        htext_best_adj.set_text(f'Best Case Effort (adj): {best_case_effort_adj}d')
        htext_best_adj.set_visible(True)
        
        hline_worst.set_data([starting_x, clicked_x], [worst_case_effort, worst_case_effort])
        hline_worst.set_visible(True)
        htext_worst.set_position((starting_x+1, worst_case_effort - 12))
        htext_worst.set_text(f'Worst Case Effort: {worst_case_effort}d')
        htext_worst.set_visible(True)
        
        hline_worst_adj.set_data([starting_x, wc_adj_fp], [worst_case_effort_adj, worst_case_effort_adj])
        hline_worst_adj.set_visible(True)
        htext_worst_adj.set_position((starting_x+1, worst_case_effort_adj + 2))
        htext_worst_adj.set_text(f'Worst Case Effort (adj): {worst_case_effort_adj}d')
        htext_worst_adj.set_visible(True)
       
        hline_likely.set_data([starting_x, clicked_x], [likely_effort, likely_effort])
        hline_likely.set_visible(True)
        htext_likely.set_position((starting_x+1, likely_effort + 2))
        htext_likely.set_text(f'Likely Effort: {likely_effort}d')
        htext_likely.set_visible(True)
        
        vline_c.set_data([clicked_x, clicked_x], [0, worst_case_effort])
        vline_c.set_visible(True)
        vtext_c.set_position((clicked_x, worst_case_effort + 3))
        vtext_c.set_text(f'{clicked_x} FP')
        vtext_c.set_visible(True) 
         
        vline_c_adj_b.set_data([bc_adj_fp, bc_adj_fp], [0, best_case_effort_adj])
        vline_c_adj_b.set_visible(True)
        vtext_c_adj_b.set_position((bc_adj_fp, best_case_effort_adj + 3))
        vtext_c_adj_b.set_text(f'{bc_adj_fp} FP')
        vtext_c_adj_b.set_visible(True)        
             
        vline_c_adj_w.set_data([wc_adj_fp, wc_adj_fp], [0, worst_case_effort_adj])
        vline_c_adj_w.set_visible(True)
        vtext_c_adj_w.set_position((wc_adj_fp, worst_case_effort_adj + 3))
        vtext_c_adj_w.set_text(f'{wc_adj_fp} FP')
        vtext_c_adj_w.set_visible(True) 
           
        # Loop from best_case_effort to worst_case_effort
        # contingency_start = (worst_case_effort - best_case_effort) * 0.9 
        
        # set the contingency at hte best case effort to be equal to the likely effort 
        contingency_start = likely_effort 
        
        # Define a decay constant for the exponential decay (adjust this for faster/slower dropoff)
        k = 0.03  # You can tweak this value to control the rate of decay

        # Function to calculate contingency using exponential decay
        def contingency_value(effort, min_effort, contingency_start, k):
            return contingency_start * np.exp(-k * (effort - min_effort))

        def calculate_contingency(effort, min_effort, max_effort, max_contingency, k=0.1):
            # Calculate the final contingency value at max_effort
            final_contingency = max_contingency * np.exp(-k * (max_effort - min_effort))
            
            # Calculate contingency using exponential decay and normalization
            contingency = (max_contingency * np.exp(-k * (effort - min_effort))) - final_contingency
            return contingency

        # Generate the effort range
        effort_range = np.arange(best_case_effort_adj, worst_case_effort_adj + 1)
        
        # Calculate contingency values for each effort value
        # contingency_values = contingency_value(effort_range, best_case_effort_adj, contingency_start, k)
        contingency_values = calculate_contingency(effort_range, best_case_effort_adj, worst_case_effort_adj, contingency_start, k)

        print(f'min contingency: {min(contingency_values)}')

        # Calculate total effort values by adding effort_range and contingency_values
        total_effort_values = effort_range + contingency_values
        
        # Get the index and value of the minimum total effort value
        min_total_index = np.argmin(total_effort_values)
        min_total_value = total_effort_values[min_total_index]
        print(f'Minimum Total Effort Value: {min_total_value} at index {min_total_index}')
        
        min_total_fp = effort_range[min_total_index]        
        hfund.set_data([0, min_total_fp], [min_total_value, min_total_value])
        vfund.set_data([min_total_fp, min_total_fp], [0, min_total_value])
        tfund.set_position((min_total_fp, min_total_value + 3))
        tfund.set_text(f'Total Funding: {round(min_total_value)}d\nContingency: {round(contingency_values[min_total_index])}d\nEffort: {round(effort_range[min_total_index])}d')
        hfund.set_visible(True)
        vfund.set_visible(True)
        tfund.set_visible(True)

        # Update the total effort line
        tline.set_data(effort_range, total_effort_values)
        tline.set_visible(True)
        cline.set_data(effort_range, contingency_values)
        cline.set_visible(True)
        bline.set_data(effort_range, effort_range)
        bline.set_visible(True)

        ax2.set_xlim(left=min(effort_range)-2, right=max(effort_range) + 2)
        ax2.set_ylim(bottom=0, top=max(total_effort_values) + 10) 
        ax2.legend()
        
        plt.draw()
        
        # Redraw the plot
        fig.canvas.draw_idle()

# Function to update annotation with new text and position
def update_annotation(x, y, text, annotation, ax):
    # Get the current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # temp_annotation = ax.annotate(text,
    #                           xy=(x, y),
    #                           xytext=(0, 0),
    #                           textcoords='offset points',
    #                           ha='left', va='center',
    #                           bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='white'))
    # plt.draw()
    
    # # Get the width of the annotation text in pixels
    # renderer = fig.canvas.get_renderer()
    
    # bbox = temp_annotation.get_window_extent(renderer=renderer)
    # text_width = bbox.width
    # text_height = bbox.height
    # print(f'w:{text_width}, h:{text_height}')

    # # Remove the temporary annotation
    # temp_annotation.remove()  

    # Update the annotation
    annotation.xy = (x, y)
    annotation.set_text(text)
    
    # print(f'x:{x}, avg:{np.average(xlim)}')
    # if x < np.average(xlim):
    #     text_width = x
    # if x < np.average(ylim):
    #     text_height = y
    
    annotation.xytext = (x, y)
    annotation.set_visible(True)
        
    plt.draw()  # Redraw the plot

# Function to update crosshair lines and annotation
def update_crosshair(event):
    # Check if the event is within the 
    if event.inaxes == ax:
        
        # Find the nearest function point and update annotation
        closest_idx = np.argmin(np.abs(function_points - event.xdata))
        closest_fp = function_points[closest_idx]
        closest_effort = effort_days[closest_idx]

        text = f'FP: {closest_fp}\nEffort: {closest_effort} days'
        update_annotation(event.xdata, event.ydata, text, annot, ax)

    elif event.inaxes == ax2:
        if len(tline.get_xdata()) == 0:
            return
        
        min_fp = np.min(tline.get_xdata())
        max_fp = np.max(tline.get_xdata())
        
        if event.xdata > max_fp or event.xdata < min_fp:
            return

        hover_fp = round(event.xdata-min_fp)
        text = f'FP: {round(event.xdata)}\nFunding: {round(tline.get_ydata()[hover_fp])}d\nContingency: {round(cline.get_ydata()[hover_fp])}d\nBudget: {round(bline.get_ydata()[hover_fp])}d'      
        update_annotation(event.xdata, event.ydata, text, annot2, ax2)
    else:
        annot.set_visible(False)
        annot2.set_visible(False)       

    # Redraw the plot
    fig.canvas.draw_idle()

# Connect the click event to the function
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect("motion_notify_event", update_crosshair)

plt.tight_layout()

# Display the plot
plt.show()
