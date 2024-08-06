import numpy as np
import matplotlib.pyplot as plt
import os

# Parámetros del problema
fabrication_hours = [2.5, 1.5, 2.75, 2]
finishing_hours = [3.5, 3, 3, 2]
profit = [375, 275, 475, 325]

max_fabrication_hours = 640
max_finishing_hours = 960

# Parámetros PSO (A. Parámetros del algoritmo)
num_particles = 20
max_iterations = 50
c1 = c2 = 1.4944
w = 0.6

# Inicialización de partículas
particles = np.random.randint(0, 100, (num_particles, 4))
velocities = np.random.randn(num_particles, 4)
personal_best_positions = np.copy(particles)
personal_best_scores = np.full(num_particles, -np.inf)
global_best_position = None
global_best_score = -np.inf

# Función objetivo
def objective_function(particles):
    profits = np.dot(particles, profit)
    fab_hours = np.dot(particles, fabrication_hours)
    fin_hours = np.dot(particles, finishing_hours)
    
    # Penalizaciones por violar restricciones
    penalty = np.zeros_like(profits, dtype=float)  # Aseguramos que el array sea de tipo float
    penalty[fab_hours > max_fabrication_hours] = np.inf
    penalty[fin_hours > max_finishing_hours] = np.inf
    
    return profits - penalty

# PSO loop
gbest_history = []

for iteration in range(max_iterations):
    scores = objective_function(particles)
    
    # Actualizar personal best
    better_scores = scores > personal_best_scores
    personal_best_scores = np.where(better_scores, scores, personal_best_scores)
    personal_best_positions = np.where(better_scores[:, np.newaxis], particles, personal_best_positions)
    
    # Actualizar global best
    best_particle_index = np.argmax(personal_best_scores)
    if personal_best_scores[best_particle_index] > global_best_score:
        global_best_score = personal_best_scores[best_particle_index]
        global_best_position = personal_best_positions[best_particle_index]
    
    gbest_history.append(global_best_score)
    
    # Actualizar velocidades y posiciones
    r1, r2 = np.random.rand(2)
    velocities = w * velocities + c1 * r1 * (personal_best_positions - particles) + c2 * r2 * (global_best_position - particles)
    particles = particles + velocities

# Resultados (B. Solución óptima encontrada y valor objetivo óptimo)
print(f"Mejor solución encontrada: {global_best_position}")
print(f"Valor objetivo óptimo: {global_best_score}")

# Gráfico (D. Gráfico de convergencia)
plt.plot(gbest_history)
plt.xlabel('Iteraciones')
plt.ylabel('Mejor valor objetivo')
plt.title('Convergencia del PSO')
plt.legend(['gbest'])

# Guardar el gráfico en una ubicación accesible
output_dir = "/mnt/data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, "PSO_convergencia.png")

plt.savefig(output_file)
plt.show()

### Generación del PDF con comentarios

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf():
    c = canvas.Canvas("./PSO_resultados.pdf", pagesize=letter)
    width, height = letter

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2.0, height - 40, "Resultados del Algoritmo PSO")

    # Descripción y parámetros (A. Parámetros del algoritmo)
    c.setFont("Helvetica", 12)
    text = """
    A. Parámetros del algoritmo:
    Número de partículas = 20
    Máximo número de iteraciones = 50
    Coeficientes de aceleración c1 = c2 = 1.4944
    Factor de inercia w = 0.6
    
    B. Solución óptima encontrada:
    """
    c.drawString(40, height - 80, text)

    # Solución óptima y valor objetivo (B. Solución óptima encontrada y valor objetivo óptimo)
    optimal_solution_text = f"Solución óptima (dominio): {global_best_position}"
    optimal_value_text = f"Valor objetivo óptimo (imagen): {global_best_score}"
    c.drawString(40, height - 180, optimal_solution_text)
    c.drawString(40, height - 200, optimal_value_text)
    
    # URL del repositorio (C. URL del repositorio)
    repo_url = "URL del repositorio del algoritmo PSO"
    c.drawString(40, height - 240, f"C. URL del repositorio: {repo_url}")

    # Gráfico (D. Gráfico de convergencia)
    c.drawString(40, height - 280, "D. Gráfico de convergencia:")
    c.drawImage(output_file, 40, height - 500, width=500, height=200)

    # Explicación de la reducción en 1 unidad del tiempo de acabado de la parte B (E. Explicación y demostración)
    explanation_text = """
    E. Explicación y demostración de la reducción en 1 unidad del tiempo de acabado de la parte B:
    """
    c.drawString(40, height - 540, explanation_text)

    # Observaciones (F. Observaciones/conclusiones)
    observations_text = """
    F. Observaciones/conclusiones:
    """
    c.drawString(40, height - 600, observations_text)

    c.save()

create_pdf()
