class Annulus:
    def __init__(self, dimension_number, period, sphere_radius=1.0):
        import itertools
        import numpy as np
        self.__dimension_number = dimension_number
        period = np.abs(np.array([per or 0 for per in period], dtype=np.float64))
        is_periodic = period != 0
        self.__coordinate_number_per_point = 3**np.sum(is_periodic)
        zero_pattern = (0,) * dimension_number
        self.__replication = period * ([zero_pattern] + [p for p in itertools.product(*[[-1,0,1] if q else [0] for q in is_periodic]) if p != zero_pattern])
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
        points = np.vstack([point + self.__replication for point in points])
        points_norm_sq = np.einsum('ij,ij->i', points, points).reshape((-1,1))
        t = 2 / (points_norm_sq * self.__normalization + 1)
        return (np.hstack((points * t, self.__sphere_radius * (1 - t))),
                np.hstack([(i,) * self.__coordinate_number_per_point for i in range(point_number)]),
                np.arange(point_number) * self.__coordinate_number_per_point)

    def points(self, coordinates):
        import numpy as np
        coordinates = np.array(coordinates).reshape((-1, self.__dimension_number+1))
        return coordinates[:,:-1] / (1 - coordinates[:,-1] / self.__sphere_radius)

    def __convert(self, points):
        import numpy as np
        return np.array(points).reshape((-1, self.__dimension_number))
