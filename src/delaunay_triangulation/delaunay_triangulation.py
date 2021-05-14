class DelaunayTriangulation:
    def __init__(self, convex_covering_space):
        self.__convex_covering_space = convex_covering_space
        self.__coordinates_to_points = None
        self.__coordinates_number = 0
        self.__points_to_coordinates = None
        self.__points = None
        self.__convex_hull = None

    def add_points(self, points, restart=False):
        import numpy as np
        coordinates, c2p, p2c = self.__convex_covering_space.coordinates(points)
        self.__points = self.__convex_covering_space.extend(self.__points, points)
        if self.__convex_hull is None:
            from scipy.spatial import ConvexHull
            if self.__convex_covering_space.infinity is not None:
                coordinates = np.vstack((self.__convex_covering_space.infinity, coordinates))
                self.__coordinates_to_points = np.hstack(([0], c2p + 1)) - 1
                self.__points_to_coordinates = p2c + 1
            else:
                self.__coordinates_to_points = c2p
                self.__points_to_coordinates = p2c
            self.__convex_hull = ConvexHull(coordinates, incremental=True)
        else:
            self.__coordinates_to_points = np.hstack((self.__coordinates_to_points, c2p + self.__points_number))
            self.__points_to_coordinates = np.hstack((self.__points_to_coordinates, p2c + self.__coordinates_number))

            self.__convex_hull.add_points(coordinates, restart)
        self.__coordinates_number = len(self.__coordinates_to_points)
        self.__points_number = len(self.__points_to_coordinates)

    @property
    def points(self):
        return self.__points

    @property
    def simplices(self):
        self.__simplices = self.__coordinates_to_points[
            self.__filter_external_simplices(
                self.__filter_infinite_simplices(
                    self.__convex_hull.simplices)
            )
        ]
        return self.__simplices

    @property
    def neighbors(self):
        pass

    def __filter_infinite_simplices(self, simplices):
        import numpy as np
        if self.__convex_covering_space.infinity is not None:
            finite_mask = np.all(simplices != 0, axis=1)
            simplices = simplices[finite_mask]
        return simplices

    def __filter_external_simplices(self, simplices):
        import numpy as np
        internal_mask = np.any(self.__points_to_coordinates[self.__coordinates_to_points[simplices]] == simplices, axis = 1)
        simplices = simplices[internal_mask]
        return simplices
