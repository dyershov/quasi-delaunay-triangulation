class Torus:
    def __init__(self, dimension_number, period, sphere_radius=1.0):
        import itertools
        import numpy as np
        self.__dimension_number = dimension_number
        self.__period = np.array(period)
        zero_pattern = (0,) * dimension_number
        self.__replication = self.__period * ([zero_pattern] + [p for p in itertools.product(*[[-1,0,1]]*dimension_number) if p != zero_pattern])
        self.__sphere_radius = sphere_radius
        self.__normalization = 1 / (sphere_radius * sphere_radius)
        self.__infinity = np.eye(N=1, M=self.__dimension_number+1, k=self.__dimension_number, dtype=np.float64).reshape((-1,))

    @property
    def infinity(self):
        return self.__infinity

    def extend(self, points_l, points_r):
        import numpy as np
        if points_l is None and points_r is None:
            return None
        if points_l is None:
            return self.__convert(points_r)
        if points_r is None:
            return self.__convert(points_l)
        return np.vstack((self.__convert(points_l), self.__convert(points_r)))

    def coordinates(self, points):
        import numpy as np
        points = self.__convert(points)
        point_number = len(points)
        points = np.vstack([point + self.__replication for point in np.mod(points, self.__period)])
        points_norm_sq = np.einsum('ij,ij->i', points, points).reshape((-1,1))
        t = 2 / (points_norm_sq * self.__normalization + 1)
        return (np.hstack((points * t, self.__sphere_radius * (1 - t))),
                np.hstack([(i,) * 3**self.__dimension_number for i in range(point_number)]),
                np.arange(point_number) * 3**self.__dimension_number)

    def points(self, coordinates):
        import numpy as np
        coordinates = np.array(coordinates).reshape((-1, self.__dimension_number+1))
        return coordinates[:,:-1] / (1 - coordinates[:,-1].reshape((-1,1)) / self.__sphere_radius)

    def __convert(self, points):
        import numpy as np
        return np.array(points).reshape((-1, self.__dimension_number))
