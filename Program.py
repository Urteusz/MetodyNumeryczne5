import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Funkcje do aproksymacji
def linear_func(x):
    return x


def abs_func(x):
    return np.abs(x)


def polynomial_func(x):
    return x ** 3 - 2 * x ** 2 + x - 5


def trig_func(x):
    return np.sin(x)


# Schemat Hornera do liczenia wartości wielomianu
def horner(coeffs, x):
    result = 0
    for c in reversed(coeffs):
        result = result * x + c
    return result


# Obliczanie współczynników wielomianu Hermite'a
def hermite_coeffs(func, degree, nodes):
    coeffs = []
    for i in range(degree + 1):
        integral, _ = quad(lambda x: func(x) * hermite_basis(x, i, nodes), -1, 1)
        coeffs.append(integral)
    return coeffs


# Baza Hermite'a
def hermite_basis(x, i, nodes):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        h0, h1 = 1, x
        for _ in range(2, i + 1):
            h0, h1 = h1, (2 * x * h1 - (i - 1) * h0) / i
        return h1


# Rysowanie wykresów
def plot_functions(x, original_func, approx_func):
    plt.figure(figsize=(10, 6))
    plt.plot(x, original_func, label='Funkcja oryginalna', color='blue')
    plt.plot(x, approx_func, label='Wielomian aproksymacyjny', color='red', linestyle='--')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Aproksymacja wielomianem Hermite\'a')
    plt.grid()
    plt.show()


# Obliczanie błędu aproksymacji
def approximation_error(original_func, approx_func):
    return np.sqrt(np.mean((np.array(original_func) - np.array(approx_func)) ** 2))


# Główna funkcja programu
def main():
    # Wybór funkcji
    print("Wybierz funkcję do aproksymacji:")
    print("1. Liniowa (x)")
    print("2. |x|")
    print("3. Wielomian (x³ - 2x² + x - 5)")
    print("4. Trygonometryczna (sin(x))")
    func_choice = int(input("Twój wybór (1-4): "))

    if func_choice == 1:
        func = linear_func
    elif func_choice == 2:
        func = abs_func
    elif func_choice == 3:
        func = polynomial_func
    elif func_choice == 4:
        func = trig_func
    else:
        print("Nieprawidłowy wybór!")
        return

    # Parametry aproksymacji
    a, b = map(float, input("Podaj przedział aproksymacji (a, b): ").split())
    degree = int(input("Podaj stopień wielomianu aproksymacyjnego: "))
    nodes = int(input("Podaj liczbę węzłów całkowania: "))

    # Generowanie przedziału i obliczanie wartości funkcji
    x = np.linspace(a, b, 1000)
    y_original = [func(xi) for xi in x]

    # Obliczanie współczynników wielomianu Hermite'a
    coeffs = hermite_coeffs(func, degree, nodes)

    # Wyznaczanie wartości wielomianu aproksymacyjnego
    y_approx = [horner(coeffs, xi) for xi in x]

    # Rysowanie wykresów
    plot_functions(x, y_original, y_approx)

    # Obliczanie błędu aproksymacji
    error = approximation_error(y_original, y_approx)
    print(f"Błąd aproksymacji: {error}")


if __name__ == "__main__":
    main()