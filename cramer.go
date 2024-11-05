package main

import (
    "bufio"
    "fmt"
    "os"
    "strconv"
    "strings"
    "sync"

    "gonum.org/v1/gonum/mat"
)

// loadMatrixFromFile загружает матрицу A и вектор b из файла.
func loadMatrixFromFile(filename string) (*mat.Dense, *mat.VecDense, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, nil, fmt.Errorf("файл '%s' не найден: %v", filename, err)
    }
    defer file.Close()

    var data [][]float64
    var b []float64

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := scanner.Text()
        vals := strings.Fields(line)
        var row []float64

        for _, val := range vals[:len(vals)-1] {
            if num, err := strconv.ParseFloat(val, 64); err == nil {
                row = append(row, num)
            }
        }
        if len(row) > 0 {
            data = append(data, row)
            if lastVal, err := strconv.ParseFloat(vals[len(vals)-1], 64); err == nil {
                b = append(b, lastVal)
            }
        }
    }

    if err := scanner.Err(); err != nil {
        return nil, nil, err
    }

    A := mat.NewDense(len(data), len(data[0]), nil)
    for i, row := range data {
        for j, val := range row {
            A.Set(i, j, val)
        }
    }

    bVec := mat.NewVecDense(len(b), nil)
    for i, val := range b {
        bVec.SetVec(i, val)
    }

    if A.RawMatrix().Rows != A.RawMatrix().Cols {
        return nil, nil, fmt.Errorf("матрица A должна быть квадратной")
    }

    return A, bVec, nil
}

// cramerSolve решает систему уравнений методом Крамера.
func cramerSolve(A *mat.Dense, b *mat.VecDense) (*mat.VecDense, error) {
    n := A.RawMatrix().Cols
    detA := mat.Det(A)

    if detA == 0 {
        return nil, fmt.Errorf("определитель матрицы равен нулю, система не имеет единственного решения")
    }

    x := mat.NewVecDense(n, nil)
    var wg sync.WaitGroup
    var mu sync.Mutex

    for i := 0; i < n; i++ {
        wg.Add(1)
        go func(i int) {
            defer wg.Done()
            Ai := mat.NewDense(n, n, nil)
            Ai.Copy(A)
            for j := 0; j < n; j++ {
                Ai.Set(j, i, b.AtVec(j))
            }
            detAi := mat.Det(Ai)
            mu.Lock() // Блокировка для безопасного доступа к x
            x.SetVec(i, detAi/detA)
            mu.Unlock()
        }(i)
    }

    wg.Wait() // Ожидаем завершения всех горутин
    return x, nil
}

// main основной блок
func main() {
    A, b, err := loadMatrixFromFile("matrix.txt")
    if err != nil {
        fmt.Println("Ошибка:", err)
        return
    }

    solution, err := cramerSolve(A, b)
    if err != nil {
        fmt.Println("Ошибка:", err)
        return
    }

    fmt.Println("Решение СЛАУ:")
    fmt.Println(mat.Formatted(solution))
}
