import numpy as np
import time
import matplotlib.pyplot as plt
from sympy import Matrix

begin = time.time()
#filename = input("Enter the filename for reading the data\n")
filename = 'marschner_lobb_41x41x41_uint8.raw.txt'

dim_x, dim_y, dim_z = 41, 41, 41
A = np.fromfile(filename, dtype='uint8', sep="")
A = A.reshape((dim_x, dim_y, dim_z))

ax = plt.axes(projection='3d')
ax.grid()
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

scalar_value = input("Enter the scalar value\n")
scalar_value = int(scalar_value)
v = 0
e = 0
f = 0
vertices = []
edges = []
faces = []
x = []
y = []
z = []

for i in range(dim_x):
    for j in range(dim_y):
        for k in range(dim_z):
            if (A[i, j, k] <= scalar_value):
                v+=1
                vertices.append([i, j, k])

for i in range(dim_x-1):
    for j in range(dim_y-1):
        for k in range(dim_z-1):
                if (A[i, j, k] <= scalar_value):
                    if(A[i+1, j, k] <= scalar_value):
                        e+=1
                        x.append(i)
                        y.append(j)
                        z.append(k)
                    if(A[i, j+1, k] <= scalar_value):
                        e+=1
                        x.append(i)
                        y.append(j)
                        z.append(k)
                    if(A[i, j, k+1] <= scalar_value):
                        e+=1
                        x.append(i)
                        y.append(j)
                        z.append(k)
                    if(A[i+1, j+1, k] <= scalar_value):
                        e+=1
                        x.append(i)
                        y.append(j)
                        z.append(k)
                    if(A[i+1, j, k+1] <= scalar_value):
                        e+=1
                        x.append(i)
                        y.append(j)
                        z.append(k)
                    if(A[i, j+1, k+1] <= scalar_value):
                        e+=1
                        x.append(i)
                        y.append(j)
                        z.append(k)
                    if(A[i+1, j+1, k+1] <= scalar_value):
                        e+=1
                        x.append(i)
                        y.append(j)
                        z.append(k)
                else:
                    x.append(0)
                    y.append(0)
                    z.append(0)


for i in range(dim_x-1):
    for j in range(dim_y-1):
        for k in range(dim_z-1):
                if (A[i, j, k] <= scalar_value):
                    if(A[i+1, j, k] <= scalar_value):
                        if(A[i+1, j+1, k] <= scalar_value):
                            f+=1
                        if(A[i+1, j, k+1] <= scalar_value):
                            f+=1
                        if(A[i+1, j+1, k+1] <= scalar_value):
                            f+=1
                    if(A[i, j+1, k] <= scalar_value):
                        if(A[i+1, j+1, k] <= scalar_value):
                            f+=1
                        if(A[i, j+1, k+1] <= scalar_value):
                            f+=1
                        if(A[i+1, j+1, k+1] <= scalar_value):
                            f+=1
                    if(A[i, j, k+1] <= scalar_value):
                        if(A[i+1, j, k+1] <= scalar_value):
                            f+=1
                        if(A[i, j+1, k+1] <= scalar_value):
                            f+=1
                        if(A[i+1, j+1, k+1] <= scalar_value):
                            f+=1

arr1 = [ [0] * int(e) for i in range(dim_x*dim_y*dim_z)]
p = 0

for i in range(dim_x-1):
    for j in range(dim_y-1):
        for k in range(dim_z-1):
                if (A[i, j, k] <= scalar_value):
                    if(A[i+1, j, k] <= scalar_value):
                        a = i+dim_x*j+dim_z*dim_y*k
                        b = (i+1)+dim_x*j+dim_z*dim_y*k
                        arr1[a][p] = -1
                        arr1[b][p] = 1
                        edges.append([a, b])
                        p+=1

                    if(A[i, j+1, k] <= scalar_value):
                        a = i+dim_x*j+dim_z*dim_y*k
                        b = i+dim_x*(j+1)+dim_z*dim_y*k
                        arr1[a][p] = -1
                        arr1[b][p] = 1
                        edges.append([a, b])
                        p+=1

                    if(A[i, j, k+1] <= scalar_value):
                        a = i+dim_x*j+dim_z*dim_y*k
                        b = i+dim_x*j+dim_z*dim_y*(k+1)
                        arr1[a][p] = -1
                        arr1[b][p] = 1
                        edges.append([a, b])
                        p+=1

                    if(A[i+1, j+1, k] <= scalar_value):
                        a = i+dim_x*j+dim_z*dim_y*k
                        b = (i+1)+dim_x*(j+1)+dim_z*dim_y*k
                        arr1[a][p] = -1
                        arr1[b][p] = 1
                        edges.append([a, b])
                        p+=1

                    if(A[i+1, j, k+1] <= scalar_value):
                        a = i+dim_x*j+dim_z*dim_y*k
                        b = (i+1)+dim_x*j+dim_z*dim_y*(k+1)
                        arr1[a][p] = -1
                        arr1[b][p] = 1
                        edges.append([a, b])
                        p+=1

                    if(A[i, j+1, k+1] <= scalar_value):
                        a = i+dim_x*j+dim_z*dim_y*k
                        b = i+dim_x*(j+1)+dim_z*dim_y*(k+1)
                        arr1[a][p] = -1
                        arr1[b][p] = 1
                        edges.append([a, b])
                        p+=1

                    if(A[i+1, j+1, k+1] <= scalar_value):
                        a = i+dim_x*j+dim_z*dim_y*k
                        b = (i+1)+dim_x*(j+1)+dim_z*dim_y*(k+1)
                        arr1[a][p] = -1
                        arr1[b][p] = 1
                        edges.append([a, b])
                        p+=1

arr2 = [ [0] * int(f) for i in range(int(e))]
q = 0
r = 0

for i in range(dim_x-1):
    for j in range(dim_y-1):
        for k in range(dim_z-1):
                if (A[i, j, k] <= scalar_value):
                    if(A[i+1, j, k] <= scalar_value):
                        if(A[i+1, j+1, k] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = (i+1)+dim_x*j+dim_y*dim_z*k
                            c = (i+1)+dim_x*(j+1)+dim_y*dim_z*k
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1

                        if(A[i+1, j, k+1] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = (i+1)+dim_x*j+dim_y*dim_z*k
                            c = (i+1)+dim_x*j+dim_y*dim_z*(k+1)
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1

                        if(A[i+1, j+1, k+1] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = (i+1)+dim_x*j+dim_y*dim_z*k
                            c = (i+1)+dim_x*(j+1)+dim_z*dim_y*(k+1)
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1

                    if(A[i, j+1, k] <= scalar_value):
                        if(A[i+1, j+1, k] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = i+dim_x*(j+1)+dim_y*dim_z*k
                            c = (i+1)+dim_x*(j+1)+dim_y*dim_z*k
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1

                        if(A[i, j+1, k+1] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = i+dim_x*(j+1)+dim_y*dim_z*k
                            c = i+dim_x*(j+1)+dim_y*dim_z*(k+1)
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1

                        if(A[i+1, j+1, k+1] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = i+dim_x*(j+1)+dim_y*dim_z*k
                            c = (i+1)+dim_x*(j+1)+dim_y*dim_z*(k+1)
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1

                    if(A[i, j, k+1] <= scalar_value):
                        if(A[i+1, j, k+1] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = i+dim_x*j+dim_y*dim_z*(k+1)
                            c = (i+1)+dim_x*j+dim_y*dim_z*(k+1)
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1

                        if(A[i, j+1, k+1] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = i+dim_x*j+dim_y*dim_z*(k+1)
                            c = i+dim_x*(j+1)+dim_y*dim_z*(k+1)
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1

                        if(A[i+1, j+1, k+1] <= scalar_value):
                            a = i+dim_x*j+dim_z*dim_y*k
                            b = i+dim_x*j+dim_y*dim_z*(k+1)
                            c = (i+1)+dim_x*(j+1)+dim_y*dim_z*(k+1)
                            arr2[edges.index([a, b])][r] = 1
                            arr2[edges.index([a, c])][r] = -1
                            arr2[edges.index([b, c])][r] = 1
                            r+=1
                        

dat = ax.scatter3D(x, y, z, c='b', cmap='Greens')
plt.show()

print("Number of vertices =", v)
print("Number of edges =", e)
print("Number of faces =", f)

rank = np.linalg.matrix_rank(arr2)
# print("Rank of Del2: " + str(rank))

D1 = Matrix(arr1)

nullity = D1.shape[1] - D1.rank()

betti_1 = nullity - rank

print("==========")
print('| \N{GREEK SMALL LETTER BETA}\N{SUBSCRIPT ONE} =',betti_1,'|')
print("==========")
end = time.time()

print("Execution time:", (end - begin), "s\n")