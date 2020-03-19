from __future__ import annotations
from typing import Union

import numpy as np

import vector


class Matrix:
    """A simple wrapper class for numpy.ndarray"""

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

    def __is_nested_list(self, data: Union[List[List[int]], List[List[float]]]) -> bool:
        """Checks if data is a nested list
        Returns True if data is a nested list; otherwise, False
        """
        return all(isinstance(item, list) for item in data)

    @property
    def data(self):
        """Returns the matrixs data"""
        return self.__data

    @property
    def shape(self):
        """Returns the shape of the matrix"""
        return self.__data.shape

    @property
    def dtype(self):
        """Returns the data type of the items in the matrix"""
        return str(self.__data.dtype)

    def __repr__(self) -> str:
        data = ''

        for row in self.data:
            data += f'\n\t{str(row)}'

        return f'Matrix(data:{data}, shape={self.data.shape}, dtype={self.dtype})'

    def __str__(self) -> str:
        return str(self.data)

    def __add__(self, other: Union[Matrix, vector.Vector]) -> Matrix:
        """Adds either:
            - Two matrices
            - A matrix and a vector
        Returns a new matrix
        """
        if type(other) is Matrix and self.shape != other.shape:
            raise ValueError("Matrices must be equal shaped")
        elif type(other) is vector.Vector and self.shape[1] != other.size:
            rows = self.shape[0]
            cols = self.shape[1]
            raise ValueError(f"Cannot add {rows}x{cols} matrix and 1x{other.size} vector")

        data_sum = self.data + other.data

        return Matrix(data=data_sum, dtype=self.dtype)
# 
#     def __sub__(self, other: Union[Matrix, vector.Vector]) -> Matrix:
#         pass
# 
#     def __mul__(self, other: Union[Matrix, vector.Vector]) -> Matrix:
#         pass

