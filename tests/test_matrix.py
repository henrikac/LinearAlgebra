# Copyright (C) 2020 Henrik A. Christensen
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from unittest import TestCase

import numpy as np

from linalg.matrix import Matrix
from linalg.vector import Vector


class MatrixTestCase(TestCase):
    def test_matrix_initialize_default(self):
        matrix = Matrix()

        self.assertIsInstance(matrix, Matrix)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.dtype, 'float32')
        self.assertTrue(np.array_equal(matrix.data, np.array([[0.0, 0.0], [0.0, 0.0]])))

    def test_matrix_initialize_dtype(self):
        matrix = Matrix(dtype=np.int32)

        self.assertIsInstance(matrix, Matrix)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.dtype, 'int32')
        self.assertTrue(np.array_equal(matrix.data, np.array([[0, 0], [0, 0]])))

    def test_matrix_initialize_shape(self):
        matrix = Matrix(rows=3, cols=3, dtype=np.int32)

        self.assertIsInstance(matrix, Matrix)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix.dtype, 'int32')
        self.assertTrue(np.array_equal(matrix.data, np.array([
                                                            [0, 0, 0],
                                                            [0, 0, 0],
                                                            [0, 0, 0]])))

    def test_matrix_minimum_rows(self):
        with self.assertRaises(ValueError):
            matrix = Matrix(rows=1, cols=2)

    def test_matrix_minimum_cols(self):
        with self.assertRaises(ValueError):
            matrix = Matrix(rows=2, cols=1)

    def test_matrix_add(self):
        data1 = [[3, 2], [4, -3], [2, 0]]
        data2 = [[-4, 5], [1, -6], [0, 1]]
        expected_data = [[-1, 7], [5, -9], [2, 1]]
        m1 = Matrix(data=data1, dtype=np.int32)
        m2 = Matrix(data=data2)

        matrix_add = m1 + m2

        self.assertIsInstance(matrix_add, Matrix)
        self.assertEqual(matrix_add.shape, (3, 2))
        self.assertEqual(matrix_add.dtype, 'int32')
        self.assertTrue(np.array_equal(matrix_add.data, np.array(expected_data)))

    def test_matrix_sub(self):
        data1 = [[3, 2], [4, -3], [2, 0]]
        data2 = [[-4, 5], [1, -6], [0, 1]]
        expected_data = [[7, -3], [3, 3], [2, -1]]
        m1 = Matrix(data=data1)
        m2 = Matrix(data=data2, dtype=np.int32)

        matrix_sub = m1 - m2

        self.assertIsInstance(matrix_sub, Matrix)
        self.assertEqual(matrix_sub.shape, (3, 2))
        self.assertEqual(matrix_sub.dtype, 'float32')
        self.assertTrue(np.array_equal(matrix_sub.data, np.array(expected_data)))

    def test_matrix_add_invalid(self):
        m1 = Matrix(data=[[1, 2], [3, 4]])
        m2 = Matrix(data=[[1, 2], [3, 4], [5, 6]])

        with self.assertRaises(ValueError):
            matrix_add = m1 + m2

    def test_matrix_sub_invalid(self):
        m1 = Matrix(data=[[1, 2], [3, 4]])
        m2 = Matrix(data=[[1, 2], [3, 4], [5, 6]])

        with self.assertRaises(ValueError):
            matrix_sub = m1 - m2

    def test_matrix_mul_matrix(self):
        m1_data = [[1, 2, 3], [4, 5, 6]]
        m2_data = [[10, 11], [20, 21], [30, 31]]
        expected_data = [[140, 146], [320, 335]]
        m1 = Matrix(data=m1_data, dtype=np.int32)
        m2 = Matrix(data=m2_data, dtype=np.int32)

        m_prod = m1 * m2

        self.assertIsInstance(m_prod, Matrix)
        self.assertEqual(m_prod.dtype, 'int32')
        self.assertEqual(m_prod.shape, (2, 2))
        self.assertTrue(np.array_equal(m_prod.data, np.array(expected_data)))

    def test_matrix_mul_vector_1(self):
        m_data = [[1, 2], [3, 4], [5, 6]]
        v_data = [7, 8]
        expected_data = [23, 53, 83]
        matrix = Matrix(data=m_data, dtype=np.int32)
        vector = Vector(data=v_data, dtype=np.int32)

        mv_prod = matrix * vector

        self.assertIsInstance(mv_prod, Vector)
        self.assertEqual(mv_prod.dtype, 'int32')
        self.assertEqual(mv_prod.size, 3)
        self.assertTrue(np.array_equal(mv_prod.data, np.array(expected_data)))
        self.assertTrue(np.array_equal(matrix.data, np.array(m_data)))
        self.assertTrue(np.array_equal(vector.data, np.array(v_data)))

    def test_matrix_mul_vector_2(self):
        m_data = [[0.8, 0.6, 0.4], [0.2, 0.4, 0.6]]
        v_data = [60, 50, 30]
        expected_data = [90, 50]
        matrix = Matrix(data=m_data)
        vector = Vector(data=v_data, dtype=np.int32)

        mv_prod = matrix * vector

        self.assertIsInstance(mv_prod, Vector)
        self.assertEqual(mv_prod.dtype, 'float32')
        self.assertEqual(mv_prod.size, 2)
        self.assertTrue(np.array_equal(mv_prod.data, np.array(expected_data)))

    def test_matrix_mul_scalar(self):
        scalar = 3
        m_data = [[3, 4, 2], [2, -3, 0]]
        expected_data = [[9, 12, 6], [6, -9, 0]]
        matrix = Matrix(data=m_data, dtype=np.int32)

        s_matrix = matrix * scalar

        self.assertIsInstance(s_matrix, Matrix)
        self.assertEqual(s_matrix.dtype, 'int32')
        self.assertEqual(s_matrix.shape, (2, 3))
        self.assertTrue(np.array_equal(s_matrix.data, np.array(expected_data)))

    def test_matrix_mul_matrix_invalid(self):
        m1_data = [[1, 2, 3], [4, 5, 6]]
        m2_data = [[6, 5, 4], [3, 2, 1]]
        m1 = Matrix(data=m1_data, dtype=np.int32)
        m2 = Matrix(data=m2_data, dtype=np.int32)

        with self.assertRaises(ValueError):
            m_prod = m1 * m2

    def test_matrix_mul_vector_invalid(self):
        m_data = [[1, 2, 3], [4, 5, 6]]
        v_data = [1, 2]
        m = Matrix(data=m_data, dtype=np.int32)
        v = Vector(data=v_data, dtype=np.int32)

        with self.assertRaises(ValueError):
            mv_prod = m * v

    def test_matrix_index(self):
        m_data = [[1, 2, 3], [4, 5, 6]]
        expected_data = [4, 5, 6]
        m = Matrix(data=m_data, dtype=np.int32)

        indexed_value = m[1]

        self.assertIsInstance(indexed_value, np.ndarray)
        self.assertTrue(np.array_equal(indexed_value, np.array(expected_data)))

    def test_matrix_index_column(self):
        m_data = [[1, 2, 3], [4, 5, 6]]
        expected_data = [2, 5]
        m = Matrix(data=m_data, dtype=np.int32)

        indexed_value = m[:,1]

        self.assertIsInstance(indexed_value, np.ndarray)
        self.assertTrue(np.array_equal(indexed_value, np.array(expected_data)))

    def test_matrix_iter(self):
        m_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        m = Matrix(data=m_data, dtype=np.int32)

        for i, item in enumerate(m):
            self.assertTrue(np.array_equal(item, np.array(m_data[i])))

    def test_matrix_len(self):
        m_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        m = Matrix(data=m_data, dtype=np.int32)

        m_len = len(m)

        self.assertEqual(m_len, len(m_data))

