import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- Конфігурація території та алгоритму ---
AREA_WIDTH, AREA_HEIGHT = 60, 40      # Розміри контрольованої території (ширина і висота)
CAMERA_RADIUS = 15                    # Радіус дії однієї камери (зона покриття)
POPULATION_SIZE = 20                  # Кількість особин (розташувань камер) у популяції
NUM_GENERATIONS = 150                 # Кількість поколінь для еволюції
MUTATION_PROBABILITY = 0.25           # Ймовірність мутації для однієї особини
min_cameras = 2                       # Мінімальна кількість камер для пошуку оптимуму
max_cameras = 10                      # Максимальна кількість камер для пошуку оптимуму

# --- Побудова сітки покриття для оцінки ---
x_values = np.linspace(0, AREA_WIDTH, 100)   # 100 точок по осі X рівномірно по території
y_values = np.linspace(0, AREA_HEIGHT, 100)  # 100 точок по осі Y
coverage_grid = np.array([[x, y] for x in x_values for y in y_values])  # Всі комбінації точок (сітка 100x100)

# --- Генерація початкової популяції ---
def generate_initial_population(num_cameras):
    """
    Створює початкове розташування камер.
    Кожна камера має випадкові координати у межах території.
    """
    return np.random.uniform([0, 0], [AREA_WIDTH, AREA_HEIGHT], size=(num_cameras, 2))

# --- Оцінка покриття з урахуванням штрафів ---
def evaluate_coverage_with_penalty(camera_positions):
    """
    Обчислює "фітнес" розташування камер з урахуванням покриття.
    Штрафи застосовуються за перекриття зон камер та неохоплені ділянки.
    """
    cover_counts = np.zeros(len(coverage_grid), dtype=int)

    # Підрахунок кількості камер, що покривають кожну точку сітки
    for cam in camera_positions:
        cover_counts += (np.linalg.norm(coverage_grid - cam, axis=1) <= CAMERA_RADIUS).astype(int)

    total_covered = np.sum(cover_counts > 0)         # Загальна кількість покритих точок
    overlap_penalty = np.sum(cover_counts > 1)       # Кількість точок, покритих більш ніж однією камерою
    uncovered_penalty = np.sum(cover_counts == 0)    # Кількість неохоплених точок

    # Фітнес = покриття - штрафи (ваги штрафів налаштовані експериментально)
    fitness = total_covered - 0.4 * overlap_penalty - 1.2 * uncovered_penalty
    return fitness

# --- Схрещення (кросовер) двох "батьків" ---
def crossover(parent_a, parent_b):
    """
    Виконує одноточкове схрещення двох розташувань камер.
    Вибирає випадкову точку і створює потомка зі сполученням частин батьків.
    """
    point = np.random.randint(1, len(parent_a))  # Випадкова точка розрізу
    return np.vstack((parent_a[:point], parent_b[point:]))

# --- Мутація розташування камер ---
def mutate(camera_positions):
    """
    З ймовірністю MUTATION_PROBABILITY змінює позицію однієї випадкової камери,
    зсув здійснюється у випадковому напрямку в межах половини радіуса дії камери.
    """
    if np.random.rand() < MUTATION_PROBABILITY:
        idx = np.random.randint(len(camera_positions))  # Випадковий індекс камери для мутації
        shift = np.random.uniform(-CAMERA_RADIUS / 2, CAMERA_RADIUS / 2, size=2)  # Випадковий зсув
        camera_positions[idx] += shift
        # Обмеження координат камер в межах області
        camera_positions[idx] = np.clip(camera_positions[idx], [0, 0], [AREA_WIDTH, AREA_HEIGHT])
    return camera_positions

# --- Основний цикл генетичної оптимізації ---
def optimize_camera_placement(num_cameras):
    """
    Виконує генетичний алгоритм для оптимізації розташування камер заданої кількості.
    Повертає найкраще знайдене розташування та початкове розташування першої особини.
    """
    population = [generate_initial_population(num_cameras) for _ in range(POPULATION_SIZE)]
    initial_population = population[0]  # Збереження початкового розташування першої особини

    for _ in range(NUM_GENERATIONS):
        # Оцінка фітнесу кожної особини
        fitness_scores = [evaluate_coverage_with_penalty(ind) for ind in population]

        # Відбір кращих 50% особин для наступного покоління
        top_half_indices = np.argsort(fitness_scores)[::-1][:POPULATION_SIZE // 2]
        parents = [population[i] for i in top_half_indices]

        new_population = parents.copy()

        # Заповнення решти популяції потомками (схрещення + мутація)
        while len(new_population) < POPULATION_SIZE:
            p_indices = np.random.choice(len(parents), 2, replace=False)
            p1 = parents[p_indices[0]].reshape(-1, 2)
            p2 = parents[p_indices[1]].reshape(-1, 2)
            child = crossover(p1, p2)
            new_population.append(mutate(child))

        population = new_population

    # Вибір найкращої особини з кінцевої популяції
    best_index = np.argmax([evaluate_coverage_with_penalty(ind) for ind in population])
    return population[best_index], initial_population

# --- Пошук оптимальної кількості камер ---
def find_optimal_number_of_cameras():
    """
    Перебирає кількість камер у діапазоні від min_cameras до max_cameras-1,
    виконує оптимізацію для кожної кількості та обирає найкращий варіант за фітнесом.
    """
    best_result = None
    best_coverage = -1
    best_num_cameras = 0

    for num_cameras in range(min_cameras, max_cameras):
        best_cameras, initial_cameras = optimize_camera_placement(num_cameras)
        fitness = evaluate_coverage_with_penalty(best_cameras)

        # Оновлення найкращого результату, якщо поточний кращий
        if fitness > best_coverage:
            best_coverage = fitness
            best_result = best_cameras
            best_num_cameras = num_cameras
            best_initial = initial_cameras

    # Повертаємо кращу кількість камер, їх розташування та початкові камери
    return best_num_cameras, best_result, best_initial if best_result is not None else (None, None, None)

# --- Запуск пошуку оптимального розташування камер ---
best_num_cameras, best_cameras, initial_cameras = find_optimal_number_of_cameras()

if best_cameras is not None:
    # --- Виведення результатів у консоль ---
    print(f"\nОптимальна кількість камер: {best_num_cameras}")
    print("\nОптимальне розташування камер (X, Y):")
    for i, cam in enumerate(best_cameras):
        print(f"Камера {i + 1}: ({cam[0]:.2f}, {cam[1]:.2f})")

    # --- Запис оптимальних координат камер у файл ---
    with open("cam_coord.txt", "w") as f:
        for cam in best_cameras:
            f.write(f"{cam[0]:.2f}, {cam[1]:.2f}\n")

    # --- Візуалізація результатів ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Відображення всієї сітки точок (потенційних зон покриття)
    ax.scatter(coverage_grid[:, 0], coverage_grid[:, 1], s=1, color='gray', alpha=0.3)

    # Відображення оптимальних позицій камер
    ax.scatter(best_cameras[:, 0], best_cameras[:, 1], color='blue', label="Optimized Cameras")

    # Відображення початкових позицій камер для порівняння
    ax.scatter(initial_cameras[:, 0], initial_cameras[:, 1], color='red', label="Initial Cameras")

    # Намалювати кола радіусом дії камер навколо оптимальних позицій
    for cam in best_cameras:
        ax.add_patch(Circle(cam, CAMERA_RADIUS, color='blue', alpha=0.1))

    ax.set_xlim(0, AREA_WIDTH)
    ax.set_ylim(0, AREA_HEIGHT)
    ax.set_title("Optimized Camera Placement with Minimal Overlap")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

else:
    # Виведення повідомлення у випадку, якщо оптимальне розташування не знайдено
    print("Оптимальне розташування камер не знайдено. Спробуйте змінити параметри "
          "(збільшити радіус дії або кількість камер. При цьому максимальна кількість камер не повинна бути "
          "меншою за мінімальну, а мінімальна — меншою за 2).")
    print("Візуалізація не виконується: оптимальне розташування не знайдено.")
