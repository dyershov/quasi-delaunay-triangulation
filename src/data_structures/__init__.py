"""
```
Data Structures package
```
"""

__all__ = [
    "sorted_list",
]


from importlib import reload

try:
    reload(sorted_list)
except NameError:
    pass

try:
    reload(euclidean)
except NameError:
    pass

try:
    reload(torus)
except NameError:
    pass

del reload


from .sorted_list import SortedList
