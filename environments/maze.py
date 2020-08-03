import random

def get_str(cell):
    return '.' if cell else ' '

def render_maze(matrix):
    for row in matrix:
        print(''.join(get_str(cell) for cell in row))

def full_matrix(n1,n2):
    return [[True]*n2 for _ in range(n1)]

def random_matrix(n1,n2):
    return [[random.choice([True, False]) for _ in range(n2)] for _ in range(n1)]

def interior(x,y, n1,n2):
    return x >= 0 or y >= 0 or x <= (n1-1) or y <= (n2-1)

def valid_neighbours(x,y, n1, n2, visited):
    neighbours = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    #print(neighbours, n1, n2)
    m = list(filter(lambda z: interior(z[0],z[1],n1,n2) and z not in visited, neighbours))
    #print(m)
    return m

def has_path(matrix, start_point, end_point, visited):
    if start_point == end_point:
        return True
    x, y = start_point
    n1, n2 = len(matrix), len(matrix[0])
    nexts = list(filter(lambda z: not matrix[z[0]][z[1]], valid_neighbours(x,y, n1, n2, visited)))
    print(nexts)
    for point in nexts:
        visited.add(point)
    return any(has_path(matrix, point, end_point, visited) for point in nexts)


# def random_maze(matrix, start_point, end_point, sparseness, visited):
#     visited.add(start_point)
#     n1, n2 = len(matrix), len(matrix[0])
#     x, y = start_point
#     if boundary(x, y, n1, n2):
#         return
#     matrix[x][y] = False
#     neighbours = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
#     possible_next = list(filter(lambda p: p not in visited, neighbours))
#     next_points = random.sample(possible_next, min(len(possible_next), sparseness))
#     print(start_point, next_points)
#     if not possible_next:
#         return
#     for point in next_points:
#         random_maze(matrix, point, end_point, sparseness, visited)

def random_maze(matrix, start_point, end_point):
    n1, n2 = len(matrix), len(matrix[0])
    x,y = start_point
    matrix[x][y] = False
    a,b = end_point
    matrix[a][b] = False
    visited = set()
    visited.add(start_point)
    while not has_path(matrix, start_point, end_point, visited):
        render_maze(matrix)
        matrix[random.randrange(1,n1-1)][random.randrange(1,n2-1)] = False

m = random_matrix(10,10)
#render_maze(m)

matrix = full_matrix(10,10)
matrix[2][2] = False
matrix[2][3] = False
matrix[2][4] = False
visited = set()
visited.add((2,2))
#random_maze(matrix, (2,2), (9,5))
#render_maze(matrix)
#print(has_path(matrix, (2,2), (2,4), visited))

from random import shuffle, randrange

def print_maze(hor, ver):
    s = ""
    for (a, b) in zip(hor, ver):
        s += ''.join(a + ['\n'] + b + ['\n'])
    return s
 
def make_maze(w = 16, h = 8):
    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    ver = [["|  "] * w + ['|'] for _ in range(h)] + [[]]
    hor = [["+--"] * w + ['+'] for _ in range(h + 1)]
    print(print_maze(hor, ver))
 
    def walk(x, y):
        vis[y][x] = 1
 
        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]: continue
            if xx == x: hor[max(y, yy)][x] = "+  "
            if yy == y: ver[y][max(x, xx)] = "   "
            walk(xx, yy)
 
    s = (randrange(w), randrange(h))
    hor[s[1]][s[0]] = "+  "
    ver[s[1]][s[0]] = " **"
    walk(s[0], s[1])
    return print_maze(hor, ver)
 
if __name__ == '__main__':
    print(make_maze())