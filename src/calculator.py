from collections import defaultdict, OrderedDict

class PatternProbabilityCalculator:
    def __init__(self, products, initial_patern_length=5, last_pattern_length=None):
        self.initial_pattern_length = initial_patern_length
        self.last_pattern_length = last_pattern_length
        self.products = products
        self.results = OrderedDict()
        self.pattern_counts = {}
        self.patterns_by_length = defaultdict(int)
        self.pattern_positions = defaultdict(list)  # New attribute to store pattern positions
    
    def calculate_next_probabilities(self):
        def extend_pattern(pattern):
            if self.last_pattern_length is not None and len(pattern) == self.last_pattern_length:
                return
            pattern_str = ','.join(map(str, pattern))
            probabilities, count = self._calculate_probabilities_for_pattern(pattern)
            
            if probabilities:
                self.results[pattern_str] = probabilities
                self.pattern_counts[pattern_str] = count
                self.patterns_by_length[len(pattern)] += 1
                
                if count > 0 and len(pattern) < len(self.products):
                    for i in range(len(self.products) - len(pattern), -1, -1):
                        if self.products[i:i + len(pattern)] == pattern:
                            pattern_index = i
                            break
                    else:
                        return
                    
                    if pattern_index > 0:
                        extended_pattern = [self.products[pattern_index - 1]] + pattern
                        extend_pattern(extended_pattern)
 
        initial_pattern = self.products[-self.initial_pattern_length:]
        extend_pattern(initial_pattern)
        
        return self.results, self.pattern_counts, self.patterns_by_length

    def _calculate_probabilities_for_pattern(self, pattern):
        pattern_length = len(pattern)
        occurrences = defaultdict(int)
        pattern_count = 0
        pattern_positions = []

        for i in range(len(self.products) - pattern_length + 1):
            current_pattern = self.products[i:i + pattern_length]
            
            if current_pattern == pattern:
                pattern_count += 1
                pattern_positions.append(i)  # Track the position of the pattern
                if i + pattern_length < len(self.products):
                    next_point = self.products[i + pattern_length]
                    occurrences[next_point] += 1
        
        total_occurrences = sum(occurrences.values())
        
        if total_occurrences == 0:
            self.pattern_positions[tuple(pattern)] = pattern_positions  # Store positions even if no occurrences found
            return OrderedDict(), pattern_count
        
        unique_elements = OrderedDict.fromkeys(self.products)
        probabilities = OrderedDict((elem, occurrences.get(elem, 0) / total_occurrences) for elem in unique_elements)
        
        self.pattern_positions[tuple(pattern)] = pattern_positions  # Store positions of the pattern
        return probabilities, pattern_count
    
    def calculate_average_probabilities(self):
        total_probs = defaultdict(float)
        count = defaultdict(int)
        
        for probabilities in self.results.values():
            for element, prob in probabilities.items():
                total_probs[element] += prob
                count[element] += 1
             
        avg_probs = {element: total_probs[element] / count[element] for element in total_probs}
        
        if not avg_probs:  # Handle empty avg_probs
            return {}, None, 0.0  # Return default values or handle the case gracefully
        highest_prob_element = max(avg_probs, key=avg_probs.get)
        return avg_probs, highest_prob_element, avg_probs[highest_prob_element]
    
    def sort_results(self):
        sorted_results = OrderedDict()
        
        for outer_key in self.results.keys():
            inner_dict = self.results[outer_key]
            sorted_inner_dict = OrderedDict(sorted(inner_dict.items()))
            sorted_results[outer_key] = sorted_inner_dict
        
        return sorted_results
    
    def find_pattern_positions(self):
        """Return the positions of each pattern in the product data."""
        return self.pattern_positions



