from src.DataProcessor import ProductDataProcessor  # type: ignore
from src.calculator import PatternProbabilityCalculator  # type: ignore

data_processor = ProductDataProcessor()

def add_product_features(features):
    # Call the add_features method in the data processor class
    data_processor.add_features(features)

def predict_values(start_number=0,initial_patern_length = 5,end_pattern_length = None,end_number=None):
    # Generate product data based on the range
    series, dozen, parity, color, column, group = data_processor.get_product_data(start=start_number, end=end_number)
    
    # List to hold predicted values
    predicted_values = []

    # Predicting for each category
    categories = {
        'Series': series,
        'Column': column,
        'Parity': parity,
        'Color': color,
        'Dozen': dozen,
        'Group': group,
    }
    
    for _, data in categories.items():
        calculator = PatternProbabilityCalculator(data ,initial_patern_length, end_pattern_length)
        _, _, _ = calculator.calculate_next_probabilities()  
        _, highest_prob_element, _ = calculator.calculate_average_probabilities()
        
        predicted_values.append(highest_prob_element)

    # Prepare the output
    # predicted_values_str = " ".join(predicted_values)
    predicted_values_str = " ".join(map(str, predicted_values))

    return predicted_values_str


def predict_continuous_values(start_number=0, end_number=None, num_predictions=5,initial_patern_length = 5,end_pattern_length = None):
    predictions = []
    correct_counts = [0] * 6  # Assuming there are 6 features to predict
    actual_values_list = []  # List to store actual values for each prediction
    current_end = end_number if end_number is not None else 0
    
    # Get the total length of the available data
    total_data_length = data_processor.get_total_data_length()
    
    for _ in range(num_predictions):
        # Check if current_end exceeds the total length of the data
        if current_end >= total_data_length:
            break
        
        # For each prediction, get fresh data up to the current number
        series, dozen, parity, color, column, group = data_processor.get_product_data(
            start=start_number, 
            end=current_end
        )
        
        # Dictionary to store category data
        categories = {
            'Series': series,
            'Column': column,
            'Parity': parity,
            'Color': color,
            'Dozen': dozen,
            'Group': group,
        }
        
        current_prediction = []
        
        # Predict for each category using only original data
        for _, category_data in categories.items():
            calculator = PatternProbabilityCalculator(category_data,initial_patern_length, end_pattern_length)
            _, _, _ = calculator.calculate_next_probabilities()
            _, highest_prob_element, _ = calculator.calculate_average_probabilities()
            current_prediction.append(highest_prob_element)
        
        # Get actual features for comparison
        actual_features = [series[-1], column[-1], parity[-1], color[-1], dozen[-1], group[-1]]
        actual_values_list.append(actual_features)
        
        # Compare predictions with actual features and update correct counts
        for j in range(6):
            if current_prediction[j] == actual_features[j]:
                correct_counts[j] += 1
        
        # Format prediction string with number
        next_number = current_end + 1
        prediction_str = f"#{next_number}: {' '.join(map(str, current_prediction))}"
        predictions.append(prediction_str)
        
        # Increment current_end for next iteration
        current_end += 1
        
    # Calculate accuracy for each feature
    accuracies = [(count / num_predictions) * 100 for count in correct_counts]
    
    # Calculate overall accuracy
    total_correct = sum(correct_counts)
    overall_accuracy = (total_correct / (num_predictions * 6)) * 100
    
    return predictions, correct_counts, actual_values_list, accuracies , overall_accuracy

if __name__ == "__main__":
    print(predict_values())
