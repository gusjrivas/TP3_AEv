import numpy as np
import matplotlib.pyplot as plt

# Definición de parámetros del problema
TIEMPO_FABRICACION = np.array([2.5, 1.5, 2.75, 2.0])
TIEMPO_ACABADO = np.array([3.5, 3.0, 3.0, 2.0])
UTILIDAD = np.array([375, 275, 475, 325])
CAPACIDAD_FABRICACION = 640
CAPACIDAD_ACABADO = 960

# Parámetros del PSO
NUM_PARTICULAS = 20
MAX_ITERACIONES = 50
C1 = 1.4944
C2 = 1.4944
W = 0.6

def funcion_objetivo(x):
    return np.dot(x, UTILIDAD)

def restricciones(x):
    fabricacion = np.dot(x, TIEMPO_FABRICACION)
    acabado = np.dot(x, TIEMPO_ACABADO)
    return (fabricacion <= CAPACIDAD_FABRICACION) and (acabado <= CAPACIDAD_ACABADO)

def inicializar_particulas():
    particulas = np.random.rand(NUM_PARTICULAS, 4) * 100
    velocidades = np.zeros((NUM_PARTICULAS, 4))
    mejor_posicion = particulas.copy()
    mejor_valor = np.array([funcion_objetivo(p) if restricciones(p) else -np.inf for p in particulas])
    return particulas, velocidades, mejor_posicion, mejor_valor

def actualizar_particulas(particulas, velocidades, mejor_posicion, mejor_valor, mejor_global):
    for i in range(NUM_PARTICULAS):
        r1, r2 = np.random.rand(2)
        velocidades[i] = (W * velocidades[i] +
                          C1 * r1 * (mejor_posicion[i] - particulas[i]) +
                          C2 * r2 * (mejor_global - particulas[i]))
        particulas[i] += velocidades[i]
        particulas[i] = np.clip(particulas[i], 0, None)  # Asegurar valores no negativos
        
        if restricciones(particulas[i]):
            valor_actual = funcion_objetivo(particulas[i])
            if valor_actual > mejor_valor[i]:
                mejor_valor[i] = valor_actual
                mejor_posicion[i] = particulas[i]
    
    return particulas, velocidades, mejor_posicion, mejor_valor

def pso():
    particulas, velocidades, mejor_posicion, mejor_valor = inicializar_particulas()
    mejor_global = mejor_posicion[np.argmax(mejor_valor)]
    mejor_global_valor = np.max(mejor_valor)
    historial_mejor_global = []

    for _ in range(MAX_ITERACIONES):
        particulas, velocidades, mejor_posicion, mejor_valor = actualizar_particulas(
            particulas, velocidades, mejor_posicion, mejor_valor, mejor_global)
        
        indice_mejor = np.argmax(mejor_valor)
        if mejor_valor[indice_mejor] > mejor_global_valor:
            mejor_global = mejor_posicion[indice_mejor]
            mejor_global_valor = mejor_valor[indice_mejor]
        
        historial_mejor_global.append(mejor_global_valor)

    return mejor_global, mejor_global_valor, historial_mejor_global

# Ejecutar el algoritmo
solucion_optima, valor_optimo, historial = pso()

# Graficar la evolución del mejor global
plt.figure(figsize=(10, 6))
plt.plot(range(1, MAX_ITERACIONES + 1), historial)
plt.xlabel('Iteraciones')
plt.ylabel('Mejor valor global')
plt.title('Evolución del mejor valor global (gbest)')
plt.grid(True)
plt.show()

print(f"Solución óptima: {solucion_optima}")
print(f"Valor óptimo: {valor_optimo}")

# Función para probar con tiempo de acabado reducido
def pso_tiempo_reducido():
    global TIEMPO_ACABADO
    TIEMPO_ACABADO[1] -= 1  # Reducir en 1 el tiempo de acabado de la parte B
    solucion, valor, _ = pso()
    TIEMPO_ACABADO[1] += 1  # Restaurar el valor original
    return solucion, valor

solucion_reducida, valor_reducido = pso_tiempo_reducido()
print(f"\nSolución con tiempo de acabado reducido: {solucion_reducida}")
print(f"Valor con tiempo de acabado reducido: {valor_reducido}")
