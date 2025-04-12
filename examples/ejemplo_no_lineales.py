from RC23104UNO import biseccion

# Ejemplo de Bisección
print("=== Método de Bisección ===")
f = lambda x: x**3 - x - 2
raiz = biseccion(f, 1, 2)
print(f"Raíz aproximada: {raiz}")
print(f"f(raíz) = {f(raiz)}")