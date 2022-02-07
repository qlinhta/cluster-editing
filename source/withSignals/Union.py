'''
Created by: Quyen Linh TA
Username: qlinhta
Date: 14/12/2021
PyCharm 2021.3.1 (Professional Edition)
Licensed to Quyen Linh TA
'''

class Union:

    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.count = [1 for i in range(n)]

    def find(self, a):
        root = a
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[a] != root:
            next_ele = self.parent[a]
            self.parent[a] = root
            a = next_ele

        return root

    def merge(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if self.count[a] < self.count[b]:
            a, b = b, a

        self.parent[b] = a
        self.count[a] += self.count[b]
        return True
