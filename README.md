# Linear Algebra

## Requirements
+ Python 3.7+

### Example (Vector):
```python
import numpy as np

from vector import Vector


vec1 = Vector(data=[1, 3, -5])  # => Vector(data=[1. 3. -5.], dtype=float32)
vec2 = Vector(size=3, dtype=np.int32)  # => Vector(data=[0 0 0], dtype=int32)
vec3 = Vector(data=[4, -2, -1], dtype=np.int32)  # => Vector(data=[4 -2 -1], dtype=int32)

vec1 + vec3  # => Vector(data=[5. 1. -6.], dtype=float32)
vec3 + vec1  # => Vector(data=[5 1 -6], dtype=int32)
vec1 * vec3  # => 3.0
vec1 * 2  # => Vector(data=[2. 6. -10.], dtype=float32)

vec1[0]  # => 1
vec1[2]  # => -5
vec1[-1]  # => -5

for item in vec3:
	print(item, end=' ')  # => 4 -2 -1
```
