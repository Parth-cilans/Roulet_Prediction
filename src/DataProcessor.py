import pandas as pd
from utils.utils import file_path

class ProductDataProcessor:
    def __init__(self):
        self.file_path = file_path
        self.df = pd.read_excel(self.file_path)
    
    def add_features(self, features=None):
        # If no features are provided, return without making changes
        if features is None:
            return
        
        # Expected features
        expected_features = ['Number','Dozen', 'Column', 'parity', 'color', 'series', 'Group']
        
        # Create a new row with the provided features
        new_row = {}
        for feature_name in expected_features:
            if feature_name in features:
                new_row[feature_name] = features[feature_name]
            else:
                new_row[feature_name] = None  # or some default value
                print(f"Warning: {feature_name} not provided.")
        
        # Convert the new row to a DataFrame
        new_row_df = pd.DataFrame([new_row])
        
        # Concatenate the new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        
        # Save the updated DataFrame back to the Excel file
        self.df.to_excel(self.file_path, index=False)
        
    
    def get_product_data(self, start=None, end=None):
        # If start and end are not provided, use the entire DataFrame
        if start is None:
            start = 0
        if end is None :
            end = len(self.df)
        if end == 0 :
            end = len(self.df)
        
        product_data = self.df[start:end]

        series = product_data['series'].tolist()
        dozen = product_data['Dozen'].tolist()
        parity = product_data['parity'].tolist()
        color = product_data['color'].tolist()
        column = product_data['Column'].tolist()
        group = product_data['Group'].tolist()

        return series, dozen, parity, color, column, group
    
    def get_total_data_length(self):
        return len(self.df)