class DelaunayTriangulation:
    def __init__(self, convex_covering_space):
        self.__convex_covering_space = convex_covering_space
        self.__d = None
        self.__n = None

        self.__coordinates_number = 0
        self.__coordinates = None
        self.__coordinates_to_points = None

        self.__points_number = 0
        self.__points = None
        self.__points_to_coordinates = None

        self.__add_infinity()

        self.__vertex_number = 0

        self.__face_number = 0
        self.__faces = None
        self.__hyperplanes = None
        self.__neighbors = None
        self.__front = None
        self.__visibility_dag = dict()
        # self.__centers = None
        # self.__radii = None

    @property
    def coordinates(self):
        return self.__coordinates

    @property
    def points(self):
        return self.__points

    @property
    def simplices(self):
        faces = self.faces
        if faces is None:
            return None
        return self.__coordinates_to_points[faces]

    @property
    def faces(self):
        import numpy as np
        if self.__faces is None:
            return None
        return self.__filter_external_simplices(self.__filter_infinite_simplices(self.__faces[list(self.__front.data),:-1]))

    @property
    def neighbors(self):
        pass

    def add_points(self, points, restart=False):
        from itertools import combinations, product
        import numpy as np

        self.__points = self.__convex_covering_space.extend(self.__points, points)

        coordinates, c2p, p2c = self.__convex_covering_space.coordinates(points)
        if self.__coordinates is None:
            self.__coordinates = coordinates
            self.__coordinates_to_points = c2p + self.__points_number
            self.__d = cordinates.shape[1]
            self.__n = self.__d + 1
        else:
            self.__coordinates = np.vstack((self.__coordinates, coordinates))
            self.__coordinates_to_points = np.hstack((self.__coordinates_to_points, c2p + self.__points_number))

        if self.__points_to_coordinates is None:
            self.__points_to_coordinates = p2c + self.__coordinates_number
        else:
            self.__points_to_coordinates = np.hstack((self.__points_to_coordinates, p2c + self.__coordinates_number))

        self.__points_number = self.__points.shape[0]
        self.__coordinates_number = self.__coordinates.shape[0]

        # print("points number %d, coordinates number %d" % (self.__points_number, self.__coordinates_number))
        # print("points to coordinates \n", self.__points_to_coordinates)
        # print("coordinates to points \n", self.__coordinates_to_points)

        if self.__coordinates.shape[0] < self.__n:
            return

        start = -len(coordinates)
        if self.__faces is None:
            self.__create_first_faces()
            start = self.__n
        for c in self.__coordinates[start:]:
            self.__add_vertex_at(c)

        assert(self.__coordinates_number == self.__coordinates.shape[0])
        assert(self.__vertex_number == self.__coordinates_number)
        assert(self.__face_number == self.__faces.shape[0])

        # print("-"*100)
        # print("face number %d, face shape (%d, %d)" % (self.__face_number, *self.__faces.shape))
        # print("faces \n", self.__faces)
        # print("neighbors \n", self.__neighbors)
        # print("front \n", self.__front.data)
        # print("visibility DAG \n", self.__visibility_dag)
        # print("coordinates \n", self.__coordinates)
        # print("hyperplanes \n", self.__hyperplanes)
        # print("="*100)

    def __create_first_faces(self):
        from data_structures import SortedList
        import numpy as np
        hyper_simplex = np.arange(self.__n, dtype=np.int64)

        hyper_orientation = 1
        hyper_coordinates = self.__coordinates[hyper_simplex]
        if np.linalg.det(np.hstack((hyper_coordinates, np.ones((self.__n, 1))))) < 0:
            hyper_orientation = -1
        # print("hyper_orientation = %d" % hyper_orientation)
        faces_orientation = np.array([1 if i % 2 == 0 else -1  for i in range(self.__n)], dtype=np.int64).reshape((-1,1)) * hyper_orientation

        self.__neighbors = np.vstack([hyper_simplex[hyper_simplex != i] for i in range(self.__n)])
        self.__faces = np.hstack((self.__neighbors, faces_orientation))

        self.__hyperplanes = np.vstack([self.__compute_face_hyperplane(face) for face in self.__faces])

        self.__front = SortedList(range(self.__n))

        self.__face_number = self.__n
        self.__vertex_number = self.__n

    def __add_vertex_at(self, coordinates):
        import numpy as np
        added_faces = []
        added_neighbors = []

        visible_faces = self.__find_all_visible_faces(coordinates)

        # print("visible faces \n", visible_faces)

        for face_index in visible_faces:
            face = self.__faces[face_index]
            face_neighbors = self.__neighbors[face_index]
            for vertex_index in range(self.__d):
                neighbor = face_neighbors[vertex_index]
                if neighbor in visible_faces:
                    continue
                next_face = list(face[:vertex_index]) + list(face[vertex_index+1:-1]) + [self.__vertex_number, -face[-1] if (self.__d - vertex_index) % 2 == 0 else face[-1]]
                added_faces.append(next_face)
                added_neighbors.append(neighbor)

                self.__neighbors[neighbor,self.__neighbors[neighbor] == face_index] = self.__face_number + len(added_faces) - 1

        added_face_indices = list(range(self.__face_number, self.__face_number + len(added_faces)))
        for face_index in visible_faces:
            self.__front.remove(face_index)
            self.__visibility_dag[face_index] = added_face_indices
        for face_index in added_face_indices:
            self.__front.insert(face_index)

        added_faces = np.array(added_faces, dtype=np.int64)
        added_neighbors = np.hstack((-np.ones((len(added_faces), self.__d-1), dtype=np.int64),
                                     np.array(added_neighbors, dtype=np.int64).reshape((-1,1))))

        # print("added_neighbors (before)\n", added_neighbors)
        self.__compute_added_neighbors(added_faces, added_neighbors)
        # print("added_neighbors (after)\n", added_neighbors)

        self.__face_number += len(added_faces)
        self.__faces = np.vstack((self.__faces, added_faces))
        added_hyperplanes = np.vstack([self.__compute_face_hyperplane(face) for face in added_faces])
        self.__hyperplanes = np.vstack((self.__hyperplanes, added_hyperplanes))
        self.__neighbors = np.vstack((self.__neighbors, added_neighbors))

        self.__vertex_number += 1

    def __find_all_visible_faces(self, coordinates, hint=None):
        import numpy as np
        affine_coordanates = np.hstack((coordinates, [1]))

        if hint is None:
            hint = self.__locate_leaf_visible_face(coordinates)

        if hint is None:
            return set()

        face_queue = [hint]
        visible_faces = set(face_queue)
        while face_queue:
            face_index = face_queue.pop()

            for next_face in self.__neighbors[face_index]:
                if next_face in visible_faces:
                    continue
                if np.dot(self.__hyperplanes[next_face], affine_coordanates) > 0:
                    continue
                visible_faces.add(next_face)
                face_queue.append(next_face)

        return visible_faces

    def __locate_leaf_visible_face(self, coordinates):
        import numpy as np
        affine_coordanates = np.hstack((coordinates, [1]))
        face_queue = list(range(self.__n))
        while face_queue:
            face_index = face_queue.pop()
            if np.dot(self.__hyperplanes[face_index], affine_coordanates) > 0:
                continue

            if face_index in self.__front:
                return face_index

            next_faces = self.__visibility_dag[face_index]

            face_queue.extend(next_faces)

        return None

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

    def __compute_added_neighbors(self, added_faces, added_neighbors):
        from itertools import combinations, product
        import numpy as np
        face_number, vertex_number = added_faces.shape
        vertex_range = np.arange(vertex_number)
        for (face_index_1, face_index_2), (vertex_index_1, vertex_index_2) in \
            product(combinations(range(face_number), 2), product(range(vertex_number - 1), repeat=2)):
            # print(face_index_1, face_index_2, vertex_index_1, vertex_index_2)
            # print(added_faces[face_index_1], added_faces[face_index_2])
            # print(added_faces[face_index_1, vertex_range != vertex_index_1][:-1])
            # print(added_faces[face_index_2, vertex_range != vertex_index_2][:-1])
            # print(added_faces[face_index_1, vertex_range != vertex_index_1][:-1] !=
            #           added_faces[face_index_2, vertex_range != vertex_index_2][:-1])
            # print('.')
            if np.any(added_faces[face_index_1, vertex_range != vertex_index_1][:-1] !=
                      added_faces[face_index_2, vertex_range != vertex_index_2][:-1]):
                continue
            added_neighbors[face_index_1, vertex_index_1] = face_index_2 + self.__face_number
            added_neighbors[face_index_2, vertex_index_2] = face_index_1 + self.__face_number

    def __compute_face_hyperplane(self, face):
        import numpy as np
        face_affine_coordinates = np.hstack((self.__coordinates[face[:-1]], np.ones((self.__d, 1))))
        # print("face_affine_coordinates \n", face_affine_coordinates)
        coordinate_range = np.arange(self.__n)
        hyperplane = np.array([np.linalg.det(face_affine_coordinates[:,coordinate_range != i]) * (1 if i % 2 == 0 else -1) for i in range(self.__n)], dtype = np.float64)
        # print("hyperplane \n", hyperplane)
        hyperplane *= face[-1]
        return hyperplane

    def __add_infinity(self):
        import numpy as np
        infinity = self.__convex_covering_space.infinity
        if infinity is None:
            return
        self.__coordinates_number = 1
        self.__coordinates = infinity.reshape((1,-1))
        self.__coordinates_to_points = np.ones(1, dtype=np.int64) * (-1)
        self.__d = self.__coordinates.shape[1]
        self.__n = self.__d + 1
