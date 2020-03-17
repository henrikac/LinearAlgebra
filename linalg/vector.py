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
    def size(self):
        """Returns the size of the vector"""
        return self.__data.size
	
    def __repr__(self):
        return f'Vector(data={str(self.data)}, dtype={str(self.data.dtype)})'

    def __str__(self):
        return str(self.data)

    def __add__(self, other: Vector) -> Vector:
        """Adds two vectors
        Returns a new vector
        """
        if self.size != other.size:
            raise ValueError("Cannot add two vector with different sizes")

        vec_sum = self.data + other.data

        return Vector(data=vec_sum)

    def __sub__(self, other: Vector) -> Vector:
        """Subtracts two vectors
        Returns a new vector
        """
        if self.size != other.size:
            raise ValueError("Cannot subtract two vectors with different sizes")

        vec_sub = self.data - other.data

        return Vector(data=vec_sub)

    def __mul__(self, other: Union[Vector, int, float]) -> Union[Vector, int, float]:
        """Takes the scalar product of two vectors if other is a vector
        Otherwise, scales self by other
        Returns the scalar product if other is a vector; otherwise, the scaled vector
        """
        if type(other) is Vector and self.size != other.size:
            raise ValueError("Cannot take the scalar product of two vectors with different sizes")

        if type(other) is Vector:
            return np.dot(self.data, other.data)

        scaled_data = self.data * other

        return Vector(data=scaled_data)

