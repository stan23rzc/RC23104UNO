import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from RC23104UNO import (
    eliminacion_gauss,
    gauss_jordan,
    crammer,
    descomposicion_lu,
    jacobi,
    gauss_seidel
)

# Ejemplo de eliminación de Gauss
print("=== Eliminación de Gauss ===")
A = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]])
b = np.array([1, -2, 0])
x = eliminacion_gauss(A, b)
print(f"Solución: {x}")
print(f"Comprobación: {np.allclose(np.dot(A, x), b)}")

# Ejemplo de Gauss-Jordan
print("\n=== Gauss-Jordan ===")
A = np.array([[4, -2, 1], [1, 1, 1], [9, 3, 1]])
b = np.array([1, 2, 3])
x = gauss_jordan(A, b)
print(f"Solución: {x}")
print(f"Comprobación: {np.allclose(np.dot(A, x), b)}")

# Ejemplo de Crammer
print("\n=== Regla de Crammer ===")
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = np.array([8, -11, -3])
x = crammer(A, b)
print(f"Solución: {x}")
print(f"Comprobación: {np.allclose(np.dot(A, x), b)}")

# Ejemplo de Descomposición LU
print("\n=== Descomposición LU ===")
A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]])
b = np.array([7.85, -19.3, 71.4])
x = descomposicion_lu(A, b)
print(f"Solución: {x}")
print(f"Comprobación: {np.allclose(np.dot(A, x), b)}")

# Ejemplo de Jacobi
print("\n=== Método de Jacobi ===")
A = np.array([[10, -1, 2], [-1, 11, -1], [2, -1, 10]])
b = np.array([6, 25, -11])
x = jacobi(A, b)
print(f"Solución: {x}")
print(f"Comprobación: {np.allclose(np.dot(A, x), b)}")

# Ejemplo de Gauss-Seidel
print("\n=== Método de Gauss-Seidel ===")
A = np.array([[16, 3], [7, -11]])
b = np.array([11, 13])
x = gauss_seidel(A, b)
print(f"Solución: {x}")
print(f"Comprobación: {np.allclose(np.dot(A, x), b)}")