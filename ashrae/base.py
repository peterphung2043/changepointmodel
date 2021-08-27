from typing import Any
from nptyping import NDArray 

AnyByAnyNDArray =  NDArray[(Any, ...), float]
NByOneNDArray = NDArray[(Any, 1,), float]  # [[1.], [2.], [3.], ...]
OneDimNDArray = NDArray[(Any,), float]  # [ 1., 2., 3., ...]


