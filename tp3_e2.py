#######################################################################
# CEIA - 16Co2024 - Algoritmos Evolutivos - TP3 - Ejercicio 2
# Gustavo J. Rivas (a1620) | Myrna L. Degano (a1618)
#######################################################################
# PSO con restricciones
#######################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


# Función objetivo a maximizar
# x = [Cant. Impresora 1, Cant. Impresora 2]
# Retorna Utilidad total diaria de ambos equipos
def f(x):
    return 500 * x[0] + 400 * x[1]

# Primera restricción
# Cada día se dispone de $127000 de capital
# 300*I1 + 400*I2 <= 127000
def g1(x):
    return 300 * x[0] + 400 * x[1] - 127000 <= 0

# Segunda restricción
# Cada día se dispone de 4270 horas de mano de obra
# 20*I1 + 10*I2 <= 4270
# El valor de I2 puede personalizarse para análisis de sensibilidad
def g2(x):
    global mo_i2
    return 20 * x[0] + mo_i2 * x[1] - 4270 <= 0

# Obtener parámetros de ejecución
def get_params():

    num_p = input("Número de partículas del enjambre (DEFAULT: 10): ").strip()
    num_p = int(num_p) if num_p else 10

    num_i = input("Número máximo de iteraciones para la optimización (DEFAULT: 80): ").strip()
    num_i = int(num_i) if num_i else 80

    c1 = input("Coeficiente de aceleración - Componente cognitivo (DEFAULT: 2): ").strip()
    c1 = float(c1) if c1 else 2.0

    c2 = input("Coeficiente de aceleración  - Componente social (DEFAULT: 2): ").strip()
    c2 = float(c2) if c2 else 2.0

    w = input("Coeficiente de inercia (DEFAULT: 0.5): ").strip()
    w = float(w) if w else 0.5

    q = input("Unidades máximas diarias (DEFAULT: 5000): ").strip()
    q = int(q) if q else 5000

    return num_p, num_i, c1, c2, w, q

# Valida que la partícula no se salga de los valores posibles
def pos(x, p):
    return (x[0] >=0 and  x[1] >=0 and x[0] <= p and x[1] <= p)


# Parámetros
n_dimensions = 2  # Dimensiones del espacio de búsqueda
result = [["Iteración #", "Cantidad Impresora 1", "Cantidad Impresora 2", "GBest - Máxima utilidad"]]
print("\n[Parámetros de ejecución del algoritmo]")
n_particles, max_iterations, c1, c2, w, max_q = get_params()

# Parámetro adicional para análisis de sensibilidad
print("\n[Datos para análisis de sensibilidad]")
mo_i2 = input("Cantidad de horas de mano de obra para Impresora 2 (DEFAULT: 10): ").strip()
mo_i2 = int(mo_i2) if mo_i2 else 10

# Inicialización de partículas
x = np.zeros((n_particles, n_dimensions))  # posiciones
v = np.zeros((n_particles, n_dimensions))  # velocidades
pbest = np.zeros((n_particles, n_dimensions))  # matriz personal best x partícula
pbest_fit = np.zeros(n_particles)  # vector para las mejores aptitudes personales
gbest = np.zeros(n_dimensions)  # mejor solución global
gbest_fit = 0  # mejor aptitud global

# Inicializacion de partículas factibles
for i in range(n_particles):
    while True:
        # posición inicial aleatoria
        # se asumen valores no enteros, porque podría fabricarse
        # una unidad parcialmente un día y continuarse al día siguiente
        x[i] = np.random.uniform(0, max_q, n_dimensions)

        if g1(x[i]) and g2(x[i]):  # si cumple las restricciones
            break  # Partícula dentro del espacio de búsqueda

    # velocidad inicial = 0
    v[i] = np.random.uniform(0, 0, n_dimensions)

    pbest[i] = x[i].copy()  # mejor valor personal = la posicion actual
    fit = f(x[i])  # aptitud de la posicion inicial
    if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
        pbest_fit[i] = fit  # se actualiza el mejor valor personal

# Optimización
for it in range(max_iterations):
    for i in range(n_particles):
        fit = f(x[i])  # aptitud de la posición actual
        # Si es mejor y cumple las restricciones
        if fit > pbest_fit[i] and g1(x[i]) and g2(x[i]):
            pbest_fit[i] = fit  # mejor aptitud personal
            pbest[i] = x[i].copy()  # mejor posición personal
            # si la nueva aptitud es mejor que la mejor global
            if fit > gbest_fit:
                gbest_fit = fit  # mejor aptitud global
                gbest = x[i].copy()  # mejor posición global

        # actualización de la velocidad y posición de la partícula
        v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
        x[i] += v[i]

        # Si la nueva posición no es válida,
        if not (g1(x[i]) and g2(x[i]) and pos(x[i], max_q)):
            # revertir a la mejor posición personal
            x[i] = pbest[i].copy()

    result.append([it+1, round(gbest[0], 2), round(gbest[1], 2), gbest_fit])

# Resultados
print(tabulate(result, headers="firstrow", tablefmt="grid", floatfmt=".2f"))
print(f"\nUtilidad máxima: ${round(gbest_fit, 2)}")
print(f"\nCantidades diarias: Impresora 1 = {round(gbest[0], 2)}, Impresora 2 = {round(gbest[1], 2)}")

# Gráfico de gbest por iteración
x = (np.array(result))[1:, 0]  # Eje de abscisas (X) - Iteraciones
y = (np.array(result))[1:, 3] # Eje de ordenadas (Y) - GBest
y = np.asarray(y, dtype=float)
plt.figure(figsize=(20, 10))
x_last = x[-1]
y_last = y[-1]
x_rest = x[:-1]
y_rest = y[:-1]
plt.plot(x_rest, y_rest, marker='o',  markersize=3, linestyle='-', color='blue', label='gbest')
plt.plot(x_last, y_last, marker='o',  markersize=5, linestyle='', color='red', label=f'Máxima utilidad\nluego de {max_iterations} iteraciones\n$ {round(gbest_fit, 2)}')
plt.xlabel('Iteración #')
plt.ylabel('GBest (Máxima utilidad en $)')
plt.xscale('linear')
plt.xticks(ticks=range(0, max_iterations + 1, 10))
plt.legend(loc='center right', frameon=True, edgecolor='black')
plt.title('Mejor valor por iteración')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()



