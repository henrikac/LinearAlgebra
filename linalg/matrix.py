from __future__ import annotations
from typing import Union

import numpy as np


class Matrix:
    def __init__(self,
            data: Union[np.ndarray, List[List[int]], List[List[float]]] = None,
            dtype: np.dtype = np.float32,
            rows: int = 2,
            cols: int = 2) -> None:
        self.__data = self.__set_data(data, dtype, rows, cols)

    def __set_data(self,
            data: Union[np.ndarray, List[List[int]], List[List[float]]],
            dtype: np.dtype,
            rows: int,
            cols: int) -> np.ndarray:
        """Sets the data of the matrix"""
        if data is None:
            if rows < 2 or cols < 2:
                raise ValueError("Minimum rank for a matrix is 2")

            return np.zeros((rows, cols), dtype=dtype)

        if type(data) is np.ndarray and data.shape < (2, 2):
            raise ValueError("Minimum rank for a matrix is 2")
        elif type(data) is list and not self.__is_nested_list(data):
            raise ValueError("Data passed to matrix is invalid")

        return np.array(data, dtype=dtype)

    def __is_nested_list(data: Union[List[List[int]], List[List[float]]]) -> bool:
        """Checks if data is a nested list
        Returns True if data is a nested list; otherwise, False
        """
        return all(isinstance(item, list) for item in data)

#     def __repr__(self):
#         pass
# 
#     def __str__(self):
#         pass
# 
#     def __add__(self, other: Union[Matrix, Vector]) -> Matrix:
#         pass
# 
#     def __sub__(self, other: Union[Matrix, Vector]) -> Matrix:
#         pass
# 
#     def __mul__(self, other: Union[Matrix, Vector]) -> Matrix:
#         pass

