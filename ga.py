import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- Конфігурація території та алгоритму ---
AREA_WIDTH, AREA_HEIGHT = 10, 10      # Розміри зони
CAMERA_RADIUS = 15                    # Радіус дії камери
POPULATION_SIZE = 20                  # Кількість особин у популяції
NUM_GENERATIONS = 150                 # Кількість поколінь
MUTATION_PROBABILITY = 0.25           # Ймовірність мутації
MIN_CAMERAS = 3                       # Мінімальна кількість камер
MAX_CAMERAS = 4                       # Максимальна кількість камер

# --- Побудова сітки покриття ---
x_values = np.linspace(0, AREA_WIDTH, 100)
y_values = np.linspace(0, AREA_HEIGHT, 100)
coverage_grid = np.array([[x, y] for x in x_values for y in y_values])

# --- Генерація початкової популяції ---
def generate_initial_population(num_cameras):
    return np.random.uniform([0, 0], [AREA_WIDTH, AREA_HEIGHT], size=(num_cameras, 2))

# --- Оцінка покриття з урахуванням штрафів ---
def evaluate_coverage_with_penalty(camera_positions):
    cover_counts = np.zeros(len(coverage_grid), dtype=int)
    for cam in camera_positions:
        cover_counts += np.linalg.norm(coverage_grid - cam, axis=1) <= CAMERA_RADIUS

    total_covered = np.sum(cover_counts > 0)
    overlap_count = np.sum(cover_counts > 1)
    uncovered_count = np.sum(cover_counts == 0)

    fitness = total_covered - 0.4 * overlap_count - 1.2 * uncovered_count

    # Повертаємо також загальну кількість покритих точок і кількість перекритих точок
    return fitness, total_covered, overlap_count

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
    if num_cameras < MIN_CAMERAS:
        raise ValueError(f"Кількість камер не може бути меншою за {MIN_CAMERAS}")
    
    population = [generate_initial_population(num_cameras) for _ in range(POPULATION_SIZE)]
    initial_population = population[0]  # Зберігаємо тільки початкові камери

    for _ in range(NUM_GENERATIONS):
        fitness_scores = [evaluate_coverage_with_penalty(ind)[0] for ind in population]
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

    best_index = np.argmax([evaluate_coverage_with_penalty(ind)[0] for ind in population])
    return population[best_index], initial_population

# --- Пошук оптимальної кількості камер ---
def find_optimal_number_of_cameras():
    best_result = None
    best_coverage = -1
    best_num_cameras = MIN_CAMERAS
    best_total_covered = 0
    best_overlap_count = 0
    best_initial_cameras = None

    # Спробуємо різні варіанти кількості камер
    for num_cameras in range(MIN_CAMERAS, MAX_CAMERAS + 1):
        best_cameras, initial_cameras = optimize_camera_placement(num_cameras)
        fitness, total_covered, overlap_count = evaluate_coverage_with_penalty(best_cameras)

        # Порівнюємо з попереднім найкращим результатом
        if fitness > best_coverage:
            best_coverage = fitness
            best_result = best_cameras
            best_num_cameras = num_cameras
            best_total_covered = total_covered
            best_overlap_count = overlap_count
            best_initial_cameras = initial_cameras

    # Обчислюємо процент покриття і процент перекриття
    total_points = len(coverage_grid)
    coverage_percentage = (best_total_covered / total_points) * 100
    overlap_percentage = (best_overlap_count / total_points) * 100

    return best_num_cameras, best_result, best_initial_cameras, coverage_percentage, overlap_percentage

# --- Запуск пошуку оптимальної кількості камер ---
best_num_cameras, best_cameras, initial_cameras, coverage_percentage, overlap_percentage = find_optimal_number_of_cameras()

# --- Виведення результатів ---
print(f"\nОптимальна кількість камер: {best_num_cameras}")
print(f"Процент покриття території: {coverage_percentage:.2f}%")
print(f"Процент перекриття камер: {overlap_percentage:.2f}%")
print("\nОптимальне розташування камер (X, Y):")
for i, cam in enumerate(best_cameras):
    print(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})")

print("\nПочаткове розташування камер (X, Y):")
for i, cam in enumerate(initial_cameras):
    print(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})")

# --- Запис у файл ---
with open("cam_coord.txt", "w") as f:
    f.write(f"Оптимальна кількість камер: {best_num_cameras}\n")
    f.write(f"Процент покриття території: {coverage_percentage:.2f}%\n")
    f.write(f"Процент перекриття камер: {overlap_percentage:.2f}%\n")
    f.write("\nОптимальне розташування камер (X, Y):\n")
    for i, cam in enumerate(best_cameras):
        f.write(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})\n")
    f.write("\nПочаткове розташування камер (X, Y):\n")
    for i, cam in enumerate(initial_cameras):
        f.write(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})\n")

# --- Візуалізація результату ---
fig, ax = plt.subplots(figsize=(8, 6))

# Візуалізація покриття
ax.scatter(coverage_grid[:, 0], coverage_grid[:, 1], s=1, color='gray', alpha=0.3)

# Візуалізація оптимізованих камер
ax.scatter(best_cameras[:, 0], best_cameras[:, 1], color='blue', label="Optimized Cameras")

# Візуалізація початкових камер (неоптимізовані)
ax.scatter(initial_cameras[:, 0], initial_cameras[:, 1], color='red', label="Initial Cameras")

# Додавання кіл для камер
for cam in best_cameras:
    ax.add_patch(Circle(cam, CAMERA_RADIUS, color='blue', alpha=0.1))

ax.set_xlim(0, AREA_WIDTH)
ax.set_ylim(0, AREA_HEIGHT)
ax.set_title(f"Optimized Camera Placement ({best_num_cameras} Cameras)")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
