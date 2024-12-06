# Define mappings using sets for efficient intersection
DOZEN_MAP = {
    'D1': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    'D2': {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
    'D3': {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
    '0': {0}
}

COLUMN_MAP = {
    'C1': {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34},
    'C2': {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35},
    'C3': {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36},
    '0' : {0},
}

ODD_EVEN_MAP = {
    'ODD': {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35},
    'EVEN': {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36}
}

COLOR_MAP = {
    'RED': {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36},
    'BLACK': {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35},
    'GREEN': {0}
}

SERIES_MAP = {
    'A': {25, 2, 21, 4, 19, 15, 32, 0, 26, 3, 35, 12, 28, 7, 29, 18, 22},
    'B': {6, 34, 17, 1, 20, 14, 31, 9},
    'C': {33, 16, 24, 5, 10, 23, 8, 30, 11, 36, 13, 27}
}

GROUP_MAP = {
    'G1' :{22, 18, 29, 7, 28, 12, 35, 3, 26, 0, 32, 15, 19, 4, 21, 2, 25, 9, 17},
    'G2' :{33, 16, 24, 5, 10, 23, 8, 30, 11, 36, 13, 27, 6, 34, 1, 20, 14, 31},
}


file_path = 'data/Example.xlsx'



