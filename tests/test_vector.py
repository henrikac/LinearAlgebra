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

from linalg.vector import Vector


class VectorTestCase(TestCase):
    def test_vector_initialize_zero_vector_default(self):
        vec = Vector()

        self.assertIsInstance(vec, Vector)
        self.assertEqual(vec.size, 2)
        self.assertEqual(vec.dtype, 'float32')
        self.assertTrue(np.array_equal(vec.data, np.array([0.0, 0.0])))

    def test_vector_initialize_zero_vector_dtype_int(self):
        vec = Vector(dtype=np.int32)

        self.assertIsInstance(vec, Vector)
        self.assertEqual(vec.size, 2)
        self.assertEqual(vec.dtype, 'int32')
        self.assertTrue(np.array_equal(vec.data, np.array([0, 0])))

    def test_vector_initialize_zero_vector_custom_size(self):
        expected_size = 4
        vec = Vector(size=expected_size)

        self.assertIsInstance(vec, Vector)
        self.assertEqual(vec.size, expected_size)
        self.assertEqual(vec.dtype, 'float32')
        self.assertTrue(np.array_equal(vec.data, np.zeros(expected_size)))

    def test_vector_initialize_list_data(self):
        data = [1, 2, 3, 4]
        vec = Vector(data=data)

        self.assertIsInstance(vec, Vector)
        self.assertEqual(vec.size, 4)
        self.assertEqual(vec.dtype, 'float32')
        self.assertTrue(np.array_equal(vec.data, np.array(data)))

    def test_vector_initialize_list_data_dtype_int(self):
        data = [1, 2, 3]
        vec = Vector(data=data, dtype=np.int32)

        self.assertIsInstance(vec, Vector)
        self.assertEqual(vec.size, 3)
        self.assertEqual(vec.dtype, 'int32')
        self.assertTrue(np.array_equal(vec.data, np.array(data)))

    def test_vector_shape_required(self):
        data = [[1, 2, 3], [4, 5, 6]]

        with self.assertRaises(ValueError):
            vec = Vector(data=data)

    def test_vector_add(self):
        vec1 = Vector(data=[2, -4, 7], dtype=np.int32)
        vec2 = Vector(data=[5, 3, 0])

        vec_add = vec1 + vec2

        self.assertIsInstance(vec_add, Vector)
        self.assertEqual(vec_add.dtype, 'int32')
        self.assertTrue(np.array_equal(vec_add.data, np.array([7, -1, 7])))

    def test_vector_add_invalid(self):
        vec1 = Vector(data=[4, 1, 5])
        vec2 = Vector(data=[-4, 9])

        with self.assertRaises(ValueError):
            vec_add = vec1 + vec2

    def test_vector_sub(self):
        vec1 = Vector(data=[2, -4, 7], dtype=np.int32)
        vec2 = Vector(data=[5, 3, 0])

        vec_sub = vec1 - vec2

        self.assertIsInstance(vec_sub, Vector)
        self.assertEqual(vec_sub.dtype, 'int32')
        self.assertTrue(np.array_equal(vec_sub.data, np.array([-3, -7, 7])))

    def test_vector_sub_invalid(self):
        vec1 = Vector(data=[1, 2, 3 , 4])
        vec2 = Vector(data=[5, 1, 6, 2, 5, 2])

        with self.assertRaises(ValueError):
            vec_sub = vec1 - vec2

    def test_vector_mul_vector(self):
        vec1 = Vector(data=[1, 3, -5])
        vec2 = Vector(data=[4, -2, -1])

        vec_dot = vec1 * vec2

        self.assertEqual(vec_dot, 3.0)

    def test_vector_mul_int(self):
        vec = Vector(data=[1, 2, 3])
        scalar = 2

        new_vec = vec * scalar

        self.assertIsInstance(new_vec, Vector)
        self.assertEqual(new_vec.size, 3)
        self.assertEqual(new_vec.dtype, 'float32')
        self.assertTrue(np.array_equal(new_vec.data, np.array([2.0, 4.0, 6.0])))

    def test_vector_mul_float(self):
        vec = Vector(data=[1, 2, 3])
        scalar = 2.5

        new_vec = vec * scalar

        self.assertIsInstance(new_vec, Vector)
        self.assertEqual(new_vec.size, 3)
        self.assertEqual(new_vec.dtype, 'float32')
        self.assertTrue(np.array_equal(new_vec.data, np.array([2.5, 5.0, 7.5])))

    def test_vector_mul_vector_invalid(self):
        vec1 = Vector(data=[1, 2, 3])
        vec2 = Vector(data=[1, 2, 3, 4])

        with self.assertRaises(ValueError):
            vec1 * vec2

    def test_vector_rmul(self):
        vec = Vector(data=[1, 2, 3, 4], dtype=np.int32)
        expected_data = [2, 4, 6, 8]
        scalar = 2

        scaled_vec = scalar * vec

        self.assertIsInstance(scaled_vec, Vector)
        self.assertTrue(np.array_equal(scaled_vec.data, np.array(expected_data)))

    def test_vector_index(self):
        vec = Vector(data=[1, 2, 3], dtype=np.int32)

        self.assertEqual(vec[0], 1)
        self.assertEqual(vec[2], 3)

    def test_vector_negative_index(self):
        vec = Vector(data=[1, 2, 3, 4, 5], dtype=np.int32)

        self.assertEqual(vec[-1], 5)

    def test_vector_too_low_index(self):
        vec = Vector(data=[1, 2, 3], dtype=np.int32)

        with self.assertRaises(IndexError):
            idx = vec[-4]

    def test_vector_too_high_index(self):
        vec = Vector(data=[1, 2, 3], dtype=np.int32)

        with self.assertRaises(IndexError):
            idx = vec[3]

    def test_vector_len(self):
        vec = Vector(data=[1, 2, 3])
        expected_len = 3

        actual_len = len(vec)

        self.assertEqual(actual_len, expected_len)

    def test_vector_iter(self):
        data = [1, 2, 3, 4, 5]
        vec = Vector(data=data, dtype=np.int32)

        for i, item in enumerate(data):
            self.assertEqual(vec[i], item)

