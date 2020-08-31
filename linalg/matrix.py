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
                raise ValueError("Matrices has to be at least 2x2")

            return np.zeros((rows, cols), dtype=dtype)

        if type(data) is np.ndarray and data.shape < (2, 2):
            raise ValueError("Matrices has to be at least 2x2")
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

    def __add__(self, other: Matrix) -> Matrix:
        """Adds two matrices and returns a new matrix"""
        if self.shape != other.shape:
            raise ValueError("Matrices must be equal shaped")

        data_sum = self.data + other.data

        return Matrix(data=data_sum, dtype=self.dtype)

    def __sub__(self, other: Matrix) -> Matrix:
        """Subtracts two matrices and returns a new matrix"""
        if self.shape != other.shape:
            raise ValueError("Matrices must be equal shaped")

        data_sum = self.data - other.data

        return Matrix(data=data_sum, dtype=self.dtype)

    def __mul__(self, other: Union[Matrix, vector.Vector, int, float]) -> Union[Matrix, vector.Vector]:
        """Performs:
            - Matrix-Matrix multiplication if other is a matrix
            - Matrix-vector multiplication if other is a vector
            - Scales self by other if other is an int or a float
        Returns:
            - A matrix if other is a matrix
            - A vector of other is a vector
            - The scaled matrix if other is an int or a float
        """
        if type(other) is Matrix and self.shape[1] != other.shape[0]:
            raise ValueError((f"Cannot multiply a {self.shape[0]}x{self.shape[1]} matrix"
                            f" and a {other.shape[0]}x{other.shape[1]} matrix"))
        elif type(other) is vector.Vector and self.shape[1] != other.size:
            raise ValueError(("Cannot perform matrix-vector multiplication on a "
                            f"{self.shape[0]}x{self.shape[1]} matrix and a "
                            f"{other.size}x1 vector"))

        if type(other) is Matrix:
            mul_data = self.data @ other.data

            return Matrix(data=mul_data, dtype=self.dtype)

        if type(other) is vector.Vector:
            sum_data = np.zeros(self.shape[0])
            temp_matrix = Matrix(data=self.data, dtype=self.dtype)

            for i, scalar in enumerate(other.data):
                temp_matrix.data[:,i] *= scalar
            
            for i, row in enumerate(temp_matrix.data):
                sum_data[i] = np.sum(temp_matrix.data[i])

            return vector.Vector(data=sum_data, dtype=self.dtype)

        scaled_data = self.data * other

        return Matrix(data=scaled_data, dtype=self.dtype)

