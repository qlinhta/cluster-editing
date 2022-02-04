from union import Union


class Graph:
    def __init__(self, n, edges):
        self._adj = [[] for _ in range(n)]
        self._n = n  # number of vertices
        self._m = 0  # number of edges

        # adding edges
        for e in edges:
            try:
                u, v = e
                self.add_edge(u, v)
            except TypeError:
                pass

        # sorting the edges for faster access
        self.sort_edges()

    def n(self):
        return self._n

    def m(self):
        return self._m

    # add edge between u and v vertex
    def add_edge(self, u, v):
        self._adj[u].append(v)
        self._adj[v].append(u)
        self._m += 1

    # remove edge between u and v vertex
    def remove_edge(self, u, v):
        try:
            self._adj[u].remove(v)
            self._adj[v].remove(u)
            self._m -= 1
        except ValueError:
            pass

    # sort the edges
    def sort_edges(self):
        for vertex in range(self._n):
            self._adj[vertex] = sorted(self._adj[vertex])

    # return the neighbors of u
    def neighbors(self, u):
        return self._adj[u]

    # return the number of neighbors
    def degree(self, u):
        return len(self._adj[u])

    # return check if both vertices are adjacent
    def is_adjacent(self, u, v):
        return 1 if v in self._adj[u] else 0

    # kernelization
    def remove_excess_degree_one(self):
        deleted = False
        delete_decision = [False] * self._n  # maintains the record of items to be deleted
        for vertex_u in range(self._n):
            if self.degree(vertex_u) == 1:
                continue

            # delete all but one neigbor of degree 1 of v
            delete_next = False
            for vertex_v in self._adj[vertex_u]:
                if delete_decision[vertex_v]:
                    continue

                if self.degree(vertex_v) == 1:
                    if delete_next:
                        delete_decision[vertex_v] = True
                        deleted = True
                    else:
                        delete_next = True

        self._m = 0
        for vertex_u in range(self._n):
            if delete_decision[vertex_u]:
                self._adj[vertex_u] = []
            else:
                self._adj[vertex_u] = [neighbor for neighbor in self._adj[vertex_u] if not delete_decision[neighbor]]
            self._m += len(self._adj[vertex_u])

        self._m /= 2
        return deleted

    def disjoint_neighborhoods(self, u, v):
        return self.num_intersecting_neighbors(u, v) == 0

    # ajdacency lists are assumed to be sorted in increasing order
    def num_intersecting_neighbors(self, u, v):
        return len(set(self._adj[u]).intersection(set(self._adj[v])))

    def remove_edge_disjoint_neighbors(self):
        deleted = False
        marked_vertices = [-1] * self._n

        # Mark vertices that have a degree 1 adjacent or two degree 2 that are adjacent, adjacents
        for vertex_u in range(self._n):
            if self.degree(vertex_u) == 1:
                marked_vertices[self._adj[vertex_u][0]] = vertex_u
                continue

            if self.degree(vertex_u) == 2:
                first_neighbor = self._adj[vertex_u][0]
                second_neighbor = self._adj[vertex_u][1]

                if self.degree(first_neighbor) != 2:
                    if self.degree(second_neighbor) == 2:
                        first_neighbor, second_neighbor = second_neighbor, first_neighbor
                    else:
                        continue

                #  vertex_v has degree 2
                first_nested_neighbor = self._adj[first_neighbor][0]
                second_nested_neighbor = self._adj[first_neighbor][1]

                if first_nested_neighbor == second_nested_neighbor:
                    first_nested_neighbor, second_nested_neighbor = second_nested_neighbor, first_nested_neighbor
                #  vertex_x ==vertex_u
                if second_neighbor == second_nested_neighbor:
                    marked_vertices[second_neighbor] = vertex_u

        temp = []

        for vertex_u in range(self._n):
            if marked_vertices[vertex_u] == -1:
                continue

            temp.clear()

            for vertex_v in self._adj[vertex_u]:
                if vertex_v != marked_vertices[vertex_u] and self.disjoint_neighborhoods(vertex_u, vertex_v):
                    vertex_to_deleted = self._lower_bound(self._adj[vertex_v], vertex_u)
                    if vertex_to_deleted:
                        self._adj[vertex_v].remove(vertex_to_deleted)
                        deleted = True
                else:
                    temp.append(vertex_v)

            self._adj[vertex_u] = temp

        self._m = 0
        for vertex_u in range(self._n):
            self._m += self.degree(vertex_u)
        self._m /= 2

        return deleted

    def _lower_bound(self, nums, target):
        if len(nums) > 0:
            l, r = 0, len(nums) - 1
            while l <= r:
                mid = l + (r - l) // 2
                if nums[mid] >= target:
                    r = mid - 1
                else:
                    l = mid + 1
            return nums[l - 1]
        return None

    # If 2 degree 2 vertices v,w are
    # adjacent to u,x that are not adjacent,
    # remove two non adjacent edges in this c4.
    def remove_c4(self):
        deleted = False

        for vertex_u in range(self._n):
            for vertex_v in self._adj[vertex_u]:
                if self.degree(vertex_v) != 2:
                    continue

                #  other_neighbor_x is the other neighbor of vertex_v
                other_neighbor_x = self._adj[vertex_v][0]
                if other_neighbor_x == vertex_u:
                    other_neighbor_x = self._adj[vertex_v][1]

                # vertex_u and x must be non-adjacent
                if self.is_adjacent(vertex_u, other_neighbor_x) == 1:
                    continue

                for vertex_w in self._adj[vertex_u]:
                    if self.degree(vertex_w) != 2 or vertex_w == vertex_v:
                        continue

                    other_neighbor_y = self._adj[vertex_w][0]
                    if other_neighbor_y == vertex_u:
                        other_neighbor_y = self._adj[vertex_w][1]

                    if other_neighbor_y != other_neighbor_x:
                        continue

                    self.remove_edge(vertex_u, vertex_v)
                    self.remove_edge(vertex_w, other_neighbor_x)

                    deleted = True
                    break

        self._m = 0
        for vertex_u in range(self._n):
            self._m += self.degree(vertex_u)
        self._m /= 2

        return deleted

    # If 3 degree <= 3 vertices u,v,w form a triangle
    # which is not in any diamond,
    # isolate them.
    def remove_deg3_triangles(self):
        deleted = False
        to_deleted = []

        for vertex_u in range(self._n):

            if self.degree(vertex_u) != 3:
                continue

            to_deleted.clear()

            neighbor_1 = self._adj[vertex_u][0]
            neighbor_2 = self._adj[vertex_u][1]
            neighbor_3 = self._adj[vertex_u][2]

            min_degree = min(self.degree(neighbor_1), self.degree(neighbor_2), self.degree(neighbor_3))
            if min_degree > 3:  # we need at least a degree 2 or 3 in neighbors:
                continue

            nb = self.is_adjacent(neighbor_1, neighbor_2) + \
                 self.is_adjacent(neighbor_2, neighbor_3) + \
                 self.is_adjacent(neighbor_3, neighbor_1)
            if nb == 3:  # K_4
                if self.degree(neighbor_2) == 3:
                    neighbor_1, neighbor_2 = neighbor_2, neighbor_1

                if self.degree(neighbor_3) == 3:
                    neighbor_1, neighbor_3 = neighbor_3, neighbor_1

                # neighbor_1 has degree 3
                if self.degree(neighbor_2) <= 5 and self.degree(neighbor_3) <= 5:
                    for vertex_x in self.neighbors(neighbor_2):
                        if vertex_x != neighbor_1 and vertex_x != neighbor_2 and vertex_x != vertex_u:
                            to_deleted.append((neighbor_2, vertex_x))

                    for vertex_x in self.neighbors(neighbor_3):
                        if vertex_x != neighbor_1 and vertex_x != neighbor_2 and vertex_x != vertex_u:
                            to_deleted.append((neighbor_3, vertex_x))

            elif nb == 2:  # Diamond
                if self.degree(neighbor_1) <= 3 and self.degree(neighbor_2) and self.degree(neighbor_3) <= 3:
                    if self.is_adjacent(neighbor_1, neighbor_2):
                        neighbor_1, neighbor_3 = neighbor_3, neighbor_1

                    if self.is_adjacent(neighbor_1, neighbor_2):
                        neighbor_2, neighbor_3 = neighbor_3, neighbor_2
                    # neighbor_1 and neighbor_2 are not adjacent

                    if self.num_intersecting_neighbors(neighbor_1, neighbor_2) == 2:  # not a diamond
                        for vertex_x in self.neighbors(neighbor_1):
                            if vertex_x != neighbor_3 and vertex_x != vertex_u:
                                to_deleted.append((neighbor_1, vertex_x))

                        for vertex_x in self.neighbors(neighbor_2):
                            if vertex_x != neighbor_3 and vertex_x != vertex_u:
                                to_deleted.append((neighbor_2, vertex_x))

            elif nb == 1:  # Triangle
                if self.is_adjacent(neighbor_1, neighbor_3):
                    neighbor_2, neighbor_3 = neighbor_3, neighbor_2

                if self.is_adjacent(neighbor_2, neighbor_3):
                    neighbor_1, neighbor_3 = neighbor_3, neighbor_1
                # neighbor_1 and neighbor_2 are adjacent, the others are not

                if self.degree(neighbor_1) <= 3 and \
                        self.degree(neighbor_2) <= 3 and \
                        self.num_intersecting_neighbors(neighbor_1, neighbor_2) == 1:  # not a diamond

                    to_deleted.append((vertex_u, neighbor_3))

                    for vertex_x in self.neighbors(neighbor_1):
                        if vertex_x != neighbor_2 and vertex_x != vertex_u:
                            to_deleted.append((neighbor_1, vertex_x))

                    for vertex_x in self.neighbors(neighbor_2):
                        if vertex_x != neighbor_1 and vertex_x != vertex_u:
                            to_deleted.append((neighbor_2, vertex_x))

            for x, y in to_deleted:
                self.remove_edge(x, y)
                deleted = True

        return deleted

    # If 3 degree <= 3 vertices u,v,w form a triangle
    # which is not in any diamond,
    # isolate them
    def isolate_small_complete(self, s):
        deleted = False
        out_degrees = []
        outer = []

        for vertex_u in range(self._n):
            if self.degree(vertex_u) != s - 1:
                continue

            a = 0
            for vertex_v in self._adj[vertex_u]:
                for vertex_w in self._adj[vertex_u]:
                    if vertex_v != vertex_w and self.is_adjacent(vertex_v, vertex_w) == 1:
                        a += 1
            # If neighborhood is not a clique, continue
            if a != (s - 1) * (s - 2):
                continue

            out_degrees.clear()
            out_degrees.append(self.degree(vertex_u) - s + 1)

            for vertex_v in self._adj[vertex_u]:
                out_degrees.append(self.degree(vertex_v) - s + 1)

            out_degrees = sorted(out_degrees)

            acc = 0
            acc_degrees = 0

            valid = True
            for i in range(s):
                acc += s - i - 1
                acc_degrees += out_degrees[s - i - 1]
                if acc_degrees > acc:
                    valid = False
                    break

            if not valid:
                continue

            outer.clear()
            for vertex_v in self._adj[vertex_u]:
                for vertex_w in self._adj[vertex_v]:
                    if vertex_w != vertex_u and self.is_adjacent(vertex_w, vertex_u) == 0:
                        outer.append(vertex_w)

            outer = sorted(outer)
            outer_size = len(outer)

            for i in range(outer_size - 1):
                if outer[i] == outer[i + 1]:
                    valid = False

            if not valid:
                continue

            for vertex_v in self._adj[vertex_u]:
                candidates = self._adj[vertex_v]
                for vertex_x in candidates:
                    if vertex_x != vertex_u and self.is_adjacent(vertex_x, vertex_u) == 0:
                        if self.is_adjacent(vertex_x, vertex_v) == 1:
                            deleted = True
                            self.remove_edge(vertex_x, vertex_v)
        return deleted

    def kernelize(self):
        m_removed = self._m
        i = 0
        while True:
            cont = False

            while self.remove_excess_degree_one():
                cont = True

            while self.remove_edge_disjoint_neighbors():
                cont = True

            while self.remove_c4():
                cont = True

            while self.remove_deg3_triangles():
                cont = True

            for s in range(3, 11):
                while self.isolate_small_complete(s):
                    cont = True

            i += 1

            if not cont:
                break

        m_removed -= self._m
        return m_removed

    def connected_components(self):
        uf = Union(self._n)

        for vertex_u in range(self._n):
            for vertex_v in self._adj[vertex_u]:
                uf.merge(vertex_u, vertex_v)

        cc_count = 0
        cc_id_aux = [0 for i in range(self._n)]
        for vertex_u in range(self._n):
            if uf.parent[vertex_u] == vertex_u:
                cc_id_aux[vertex_u] = cc_count
                cc_count += 1
        ccs = [[] for i in range(cc_count)]
        for vertex_u in range(self._n):
            ccs[cc_id_aux[uf.find(vertex_u)]].append(vertex_u)

        return ccs
