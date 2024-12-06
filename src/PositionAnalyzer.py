import plotly.express as px
import pandas as pd
import numpy as np

class PatternPositionAnalyzer:
    def __init__(self, pattern_positions):
        self.pattern_positions = pattern_positions

    def calculate_position_differences(self):
        """Calculate the difference between consecutive positions for each pattern."""
        position_differences = {}

        for pattern, positions in self.pattern_positions.items():
            if len(positions) > 1:
                differences = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
                position_differences[pattern] = differences
            else:
                # Add a placeholder for patterns with only one position
                position_differences[pattern] = []

        return position_differences

    def plot_difference_barcharts(self, differences_dict, bin_size=10, max_range=100):
        """Create interactive bar charts showing how many differences fall into each range."""
        figs = []  # List to hold all figures
        
        for pattern, differences in differences_dict.items():
            if not differences:
                # Create a "dummy" figure for patterns with no differences
                fig = px.bar(
                    pd.DataFrame({'Difference Range': ['No differences'], 'Count': [0]}),
                    x='Difference Range',
                    y='Count',
                    title=f'Pattern {pattern} (Only one occurrence found)'
                )
                figs.append(fig)
                continue
                
            # Rest of the plotting code remains the same
            bins = np.arange(0, max_range + bin_size, bin_size)
            bin_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)]
            bin_counts = pd.cut(differences, bins=bins, labels=bin_labels, include_lowest=True).value_counts().reset_index()
            bin_counts.columns = ['Difference Range', 'Count']
            
            fig = px.bar(bin_counts, 
                         x='Difference Range', 
                         y='Count', 
                         title=f'Position Difference Ranges for Pattern {pattern}', 
                         text='Count',
                         hover_data=['Count'],
                         color='Count',
                         color_continuous_scale=px.colors.sequential.Viridis)
            
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis_title='Difference Range', yaxis_title='Count', barmode='group')
            
            figs.append(fig)
        
        return figs
