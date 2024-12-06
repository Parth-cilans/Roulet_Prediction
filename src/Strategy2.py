
def find_next_numbers_and_features(df, n):
    # Get the number in the last row
    last_number = df['Number'].iloc[-1]
    
    # Get the last `n` rows from the dataframe
    last_n_rows = df.tail(n)
    
    # Find the indices of where the last number appears in the last `n` rows
    indices = last_n_rows[last_n_rows['Number'] == last_number].index
    
    # Initialize a list to store the next numbers and their features
    next_numbers_and_features = []
    
    # Loop through each index where the last number is found
    for idx in indices:
        # Ensure we don't go out of bounds when looking for the next row
        if (idx + 1) in df.index:
            # Get the next row after the found number
            next_row = df.iloc[idx + 1]
            
            # Append the next number and its corresponding features to the list
            next_numbers_and_features.append({
                'Next Number': next_row['Number'],
                'Dozen': next_row['Dozen'],
                'Column': next_row['Column'],
                'parity': next_row['parity'],
                'color': next_row['color'],
                'series': next_row['series'],
                'Group': next_row['Group']
            })
    
    return last_number, next_numbers_and_features


def standardize_values(df):
    features_to_check = ['Dozen', 'Column']  # You can add more features here if necessary
    
    # Standardizing formatting for each feature
    for feature in features_to_check:
        df[feature] = df[feature].str.replace(r'(\D)\s*(\d)', r'\1 \2', regex=True)  # Add space between letter and number
    
    return df

# Function to calculate the most frequent events and their percentages
def calculate_feature_percentages(df):
    most_frequent_events = {}
    
    features = ['Dozen', 'Column', 'parity', 'color', 'series', 'Group']
    
    for feature in features:
        value_counts = df[feature].value_counts(normalize=True) * 100
        highest_percentage = value_counts.max()
        
        # Find all events that have the highest percentage and check for ties
        most_frequent_events_for_feature = value_counts[value_counts == highest_percentage].index.tolist()
        
        most_frequent_events[feature] = {
            'Most Frequent Events': most_frequent_events_for_feature,
            'Percentage': highest_percentage
        }
    
    return most_frequent_events 


def get_last_n_numbers_and_frequent_features(df, n=5):
    # Select the last n rows
    last_n_rows = df.tail(n)
    
    most_frequent_features = {}
    
    # List of features to analyze
    features = ['Dozen', 'Column', 'parity', 'color', 'series' , 'Group']
    
    for feature in features:
        # Get the most frequent value(s) for each feature
        value_counts = last_n_rows[feature].value_counts(normalize=True) * 100
        highest_percentage = value_counts.max()
        
        # Find all events that have the highest percentage
        most_frequent_for_feature = value_counts[value_counts == highest_percentage].index.tolist()
        
        # Store the results in the dictionary
        most_frequent_features[feature] = {
            'Most Frequent Events': most_frequent_for_feature,
            'Percentage': highest_percentage
        }
    
    return last_n_rows, most_frequent_features