from __future__ import annotations
from typing import List, Union

import numpy as np


class Vector:
    """A simple wrapper class for numpy.array"""

    def __init__(self,
            size: int = 2,
            data: Union[np.ndarray, List[Union[int, float]]] = None,
            dtype: np.dtype = np.float32) -> None:
        self.__data = self.__set_data(size, data, dtype)

    def __set_data(self,
            size: int,
            data: Union[np.ndarray, List[Union[int, float]]],
            dtype: np.dtype) -> np.ndarray:
        """Set the vectors data
        If data is None the vector will be set to a zero vector
        """
        if data is not None and type(data) is not list and data.ndim > 1:
            raise ValueError("Vectors should be rank 1")
        elif type(data) is list and any(isinstance(item, list) for item in data):
            raise ValueError("Vectors should be rank 1")

        return np.zeros(size, dtype) if data is None else np.array(data, dtype)

    @property
    def data(self):
        """Returns the vectors data"""
        return self.__data

    @property
    def dtype(self):
        """Returns the data type of the items in the vector"""
        return str(self.data.dtype)

    @property
    def size(self):
        """Returns the size of the vector"""
        return self.__data.size
	
    def __repr__(self) -> str:
        return f'Vector(data={str(self.data)}, dtype={str(self.data.dtype)})'

    def __str__(self) -> str:
        return str(self.data)

    def __add__(self, other: Vector) -> Vector:
        """Adds two vectors and returns a new vector"""
        if self.size != other.size:
            raise ValueError("Cannot add two vectors with different sizes")

        data_sum = self.data + other.data

        return Vector(data=data_sum, dtype=self.dtype)

    def __sub__(self, other: Vector) -> Vector:
        """Subtracts two vectors and returns a new vector"""
        if self.size != other.size:
            raise ValueError("Cannot subtract two vectors with different sizes")

        data_sum = self.data - other.data

        return Vector(data=data_sum, dtype=self.dtype)

    def __mul__(self, other: Union[Vector, int, float]) -> Union[Vector, int, float]:
        """Takes the scalar product of two vectors if other is a vector
        Scales self by other
        Returns:
            - The scalar product if other is a vector
            - The scaled vector if other is an int or a float
        """
        if type(other) is Vector and self.size != other.size:
            raise ValueError("Cannot take the scalar product of two vectors with different sizes")

        if type(other) is Vector:
            return np.dot(self.data, other.data)

        scaled_data = self.data * other

        return Vector(data=scaled_data, dtype=self.dtype)

