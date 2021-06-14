class SortedList:
    def __init__(self, init = None):
        from collections import deque
        self.__data = list() if init is None else list(init)
        self.__data.sort()
        self.__data = deque(self.__data)

    def __index_of__(self, element):
        from bisect import bisect_left
        return bisect_left(self.__data, element)

    def __contains__(self, element):
        self.__index_of__element = self.__index_of__(element)
        return self.__index_of__element < len(self.__data) and \
            self.__data[self.__index_of__element] == element

    @property
    def data(self):
        from copy import copy
        return copy(self.__data)

    def insert(self, element):
        self.__data.insert(self.__index_of__(element), element)

    def remove(self, element):
        if element not in self:
            raise ValueError("Element {} is not found".format(element))
        del self.__data[self.__index_of__element]
