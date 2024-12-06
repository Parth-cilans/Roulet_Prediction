from utils.utils import SERIES_MAP, COLUMN_MAP, ODD_EVEN_MAP, COLOR_MAP, DOZEN_MAP ,GROUP_MAP


def predict_features_from_number(number: int):
    try:
        # Check if the number is valid
        if not (0 <= number <= 36):
            raise ValueError(f"Invalid number: {number}")

        # Initialize predicted features
        predicted_features = {}

        # Predict the series
        for series, numbers in SERIES_MAP.items():
            if number in numbers:
                predicted_features['series'] = series
                break

        # Predict the column
        for column, numbers in COLUMN_MAP.items():
            if number in numbers:
                predicted_features['Column'] = column
                break

        # Predict odd/even
        for odd_even, numbers in ODD_EVEN_MAP.items():
            if number in numbers:
                predicted_features['parity'] = odd_even
                break

        # Predict color
        for color, numbers in COLOR_MAP.items():
            if number in numbers:
                predicted_features['color'] = color
                break

        # Predict dozen
        for dozen, numbers in DOZEN_MAP.items():
            if number in numbers:
                predicted_features['Dozen'] = dozen
                break
            
        for group, numbers in GROUP_MAP.items():
            if number in numbers:
                predicted_features['Group'] = group
                break

        # Return the predicted features as a dictionary
        return predicted_features

    except Exception as e:
        print(f"Error in predict_features_from_number: {str(e)}")
        return {}