import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- Конфігурація території та алгоритму ---
AREA_WIDTH, AREA_HEIGHT = 60, 40      # Розміри зони
CAMERA_RADIUS = 15                    # Радіус дії камери
POPULATION_SIZE = 20                  # Кількість особин у популяції
NUM_GENERATIONS = 150                 # Кількість поколінь
MUTATION_PROBABILITY = 0.25           # Ймовірність мутації

# --- Побудова сітки покриття ---
x_values = np.linspace(0, AREA_WIDTH, 100)
y_values = np.linspace(0, AREA_HEIGHT, 100)
coverage_grid = np.array([[x, y] for x in x_values for y in y_values])

# --- Генерація початкової популяції ---
def generate_initial_population(num_cameras):
    return [
        np.random.uniform([0, 0], [AREA_WIDTH, AREA_HEIGHT], size=(num_cameras, 2))
        for _ in range(POPULATION_SIZE)
    ]

# --- Оцінка покриття з урахуванням штрафів ---
def evaluate_coverage_with_penalty(camera_positions):
    cover_counts = np.zeros(len(coverage_grid), dtype=int)
    for cam in camera_positions:
        cover_counts += np.linalg.norm(coverage_grid - cam, axis=1) <= CAMERA_RADIUS

    total_covered = np.sum(cover_counts > 0)
    overlap_penalty = np.sum(cover_counts > 1)
    uncovered_penalty = np.sum(cover_counts == 0)

    fitness = total_covered - 0.4 * overlap_penalty - 1.2 * uncovered_penalty
    return fitness

# --- Схрещення ---
def crossover(parent_a, parent_b):
    point = np.random.randint(1, len(parent_a))
    return np.vstack((parent_a[:point], parent_b[point:]))

# --- Мутація ---
def mutate(camera_positions):
    if np.random.rand() < MUTATION_PROBABILITY:
        idx = np.random.randint(len(camera_positions))
        shift = np.random.uniform(-CAMERA_RADIUS / 2, CAMERA_RADIUS / 2, size=2)
        camera_positions[idx] += shift
        camera_positions[idx] = np.clip(camera_positions[idx], [0, 0], [AREA_WIDTH, AREA_HEIGHT])
    return camera_positions

# --- Генетична оптимізація розташування камер ---
def optimize_camera_placement(num_cameras):
    population = generate_initial_population(num_cameras)
    initial_population = population.copy()

    for _ in range(NUM_GENERATIONS):
        fitness_scores = [evaluate_coverage_with_penalty(ind) for ind in population]
        top_half_indices = np.argsort(fitness_scores)[::-1][:POPULATION_SIZE // 2]
        parents = [population[i] for i in top_half_indices]

        new_population = parents.copy()
        while len(new_population) < POPULATION_SIZE:
            p_indices = np.random.choice(len(parents), 2, replace=False)
            p1 = parents[p_indices[0]].reshape(-1, 2)
            p2 = parents[p_indices[1]].reshape(-1, 2)
            child = crossover(p1, p2)
            new_population.append(mutate(child))

        population = new_population

    best_index = np.argmax([evaluate_coverage_with_penalty(ind) for ind in population])
    return population[best_index], initial_population

# --- Запуск оптимізації ---
NUM_CAMERAS = 6
best_cameras, initial_population = optimize_camera_placement(NUM_CAMERAS)

# --- Виведення координат у консоль ---
print("\nОптимальне розташування камер (X, Y):")
for i, cam in enumerate(best_cameras):
    print(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})")

print("\nПочаткове розташування (перша особина популяції):")
for i, cam in enumerate(initial_population[0]):
    print(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})")

# --- Запис у файл ---
with open("camera_positions.txt", "w") as f:
    f.write("Оптимальне розташування камер (X, Y):\n")
    for i, cam in enumerate(best_cameras):
        f.write(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})\n")

    f.write("\nПочаткове розташування (перша особина популяції):\n")
    for i, cam in enumerate(initial_population[0]):
        f.write(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})\n")

# --- Візуалізація результату ---
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(coverage_grid[:, 0], coverage_grid[:, 1], s=1, color='gray', alpha=0.3)
ax.scatter(best_cameras[:, 0], best_cameras[:, 1], color='blue', label="Optimized Cameras")

for cam in best_cameras:
    ax.add_patch(Circle(cam, CAMERA_RADIUS, color='blue', alpha=0.1))

for i, cam in enumerate(initial_population[0]):
    ax.scatter(cam[0], cam[1], color='red', alpha=0.5, label="Initial Position" if i == 0 else None)

ax.set_xlim(0, AREA_WIDTH)
ax.set_ylim(0, AREA_HEIGHT)
ax.set_title("Optimized Camera Placement with Minimal Overlap")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.show()
