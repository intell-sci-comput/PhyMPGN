from torch_geometric.data import Data
from typing import Any


class Graph(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'laplace_matrix':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)