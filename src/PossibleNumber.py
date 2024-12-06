from utils.utils import SERIES_MAP, COLUMN_MAP, ODD_EVEN_MAP, COLOR_MAP, DOZEN_MAP

def get_predicted_number(series: str, column: str, odd_even: str, color: str, dozen: str):
    try:
        # Validate inputs with clear error messages for debugging
        if series not in SERIES_MAP:
            raise ValueError(f"Invalid series: {series}")
        if column not in COLUMN_MAP:
            raise ValueError(f"Invalid column: {column}")
        if odd_even not in ODD_EVEN_MAP:
            raise ValueError(f"Invalid odd/even value: {odd_even}")
        if color not in COLOR_MAP:
            raise ValueError(f"Invalid color: {color}")
        if dozen not in DOZEN_MAP:
            raise ValueError(f"Invalid dozen: {dozen}")

        # Use set intersection to get the numbers satisfying all features
        possible_numbers = SERIES_MAP[series] & COLUMN_MAP[column] & \
                          ODD_EVEN_MAP[odd_even] & COLOR_MAP[color] & \
                          DOZEN_MAP[dozen]
        
        if not possible_numbers:
            return "Number not found"

        # Convert set to sorted list for consistent output
        return sorted(list(possible_numbers))
    
    except Exception as e:
        print(f"Error in get_predicted_number: {str(e)}")
        return "Number not found"


