#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <stdexcept>

using namespace std;

mutex mtx; // Мьютекс для синхронизации вывода

// Функция для загрузки матрицы из файла
void loadMatrixFromFile(const string& filename, vector<vector<double>>& A, vector<double>& b) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Ошибка: не удалось открыть файл.");
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        A.push_back(vector<double>(row.begin(), row.end() - 1)); // Все столбцы кроме последнего
        b.push_back(row.back()); // Последний элемент как b
    }
}

// Функция для вычисления определителя матрицы
double determinant(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    if (n == 1) {
        return matrix[0][0];
    }
    if (n == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    double det = 0.0;
    for (int i = 0; i < n; ++i) {
        // Создание матрицы для Cofactor
        vector<vector<double>> subMatrix(n - 1, vector<double>(n - 1));
        for (int j = 1; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (k < i) {
                    subMatrix[j - 1][k] = matrix[j][k];
                } else if (k > i) {
                    subMatrix[j - 1][k - 1] = matrix[j][k];
                }
            }
        }
        det += (i % 2 == 0 ? 1 : -1) * matrix[0][i] * determinant(subMatrix);
    }
    return det;
}

// Функция для вычисления определителя для i-й переменной
double computeDeterminantForX(const vector<vector<double>>& A, const vector<double>& b, int i, double det_A) {
    // Создание матрицы Ai, заменяем i-й столбец на вектор b
    vector<vector<double>> Ai = A;
    for (int j = 0; j < A.size(); ++j) {
        Ai[j][i] = b[j];
    }
    return determinant(Ai) / det_A;
}

int main() {
    vector<vector<double>> A;
    vector<double> b;

    try {
        loadMatrixFromFile("matrix.txt", A, b);

        int n = A.size();
        // Проверка на квадратность матрицы
        for (const auto& row : A) {
            if (row.size() != n) {
                throw runtime_error("Ошибка: матрица A должна быть квадратной.");
            }
        }

        double det_A = determinant(A);
        if (det_A == 0) {
            throw runtime_error("Ошибка: определитель матрицы равен нулю, система не имеет единственного решения.");
        }

        vector<double> solutions(n);
        vector<thread> threads;

        // Запускаем потоки для вычисления x_i
        for (int i = 0; i < n; ++i) {
            threads.emplace_back([&, i]() {
                double result = computeDeterminantForX(A, b, i, det_A);
                lock_guard<mutex> lock(mtx); // Блокировка для безопасного доступа к shared data
                solutions[i] = result;
            });
        }

        for (auto& th : threads) {
            th.join(); // Ожидаем завершения всех потоков
        }

        cout << "Решение СЛАУ: ";
        for (double x : solutions) {
            cout << x << " ";
        }
        cout << endl;

    } catch (const exception& e) {
        cout << "Ошибка: " << e.what() << endl;
    }

    return 0;
}
