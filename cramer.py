import numpy as np
from multiprocessing import Pool
import os

def load_matrix_from_file(filename):
    """
    Загружает матрицу A и вектор b из файла.
    Последний столбец файла считается вектором b, а остальные — матрицей A.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл '{filename}' не найден.")
    
    # Загружаем данные из файла
    data = np.loadtxt(filename)
    
    # Разделяем на матрицу A и вектор b
    A = data[:, :-1]  # Все столбцы, кроме последнего, для матрицы A
    b = data[:, -1]   # Последний столбец для вектора b

    # Проверяем, что матрица A квадратная
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной.")
    
    return A, b

def compute_determinant(args):
    """
    Вычисляет определитель для i-й переменной, заменяя i-й столбец матрицы A на вектор b.
    Аргументы передаются в виде кортежа для удобства использования с multiprocessing.
    """
    A, b, i, det_A = args
    A_i = np.copy(A)
    A_i[:, i] = b
    det_A_i = np.linalg.det(A_i)
    return det_A_i / det_A

def cramer_solve_parallel(A, b):
    """
    Решение системы линейных уравнений Ax = b методом Крамера с параллельными вычислениями.

    Аргументы:
    - A: квадратная матрица коэффициентов
    - b: вектор значений

    Возвращает:
    - Вектор x с решениями, если определитель системы ненулевой.
    """
    n = len(b)
    det_A = np.linalg.det(A)

    if det_A == 0:
        raise ValueError("Определитель матрицы равен нулю, система не имеет единственного решения.")

    # Создаем список аргументов для параллельного вычисления
    args = [(A, b, i, det_A) for i in range(n)]

    # Параллельное вычисление с использованием multiprocessing.Pool
    with Pool() as pool:
        x = pool.map(compute_determinant, args)

    return x

# Основной блок
if __name__ == "__main__":
    try:
        # Загружаем матрицу A и вектор b из файла
        A, b = load_matrix_from_file("matrix.txt")

        # Запуск решения
        solution = cramer_solve_parallel(A, b)
        print("Решение СЛАУ:", np.round(solution, decimals=6))
    except (FileNotFoundError, ValueError) as e:
        print("Ошибка:", e)
