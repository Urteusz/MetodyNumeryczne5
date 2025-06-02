import math
import numpy as np
import matplotlib.pyplot as plt

# Przykładowe funkcje do aproksymacji
def f1(x): return x
def f2(x): return np.abs(x)
def f3(x): return x**3 - 2 * x + 1
def f4(x): return np.sin(x)
def f5(x): return np.sin(x**2)

# Słownik dostępnych funkcji z opisami
functions = {
    '1': ("x", f1),
    '2': ("|x|", f2),
    '3': ("x^3 - 2x + 1", f3),
    '4': ("sin(x)", f4),
    '5': ("sin(x^2)", f5)
}

# Generuje wielomiany Hermite'a (fizyków) do stopnia n włącznie
def hermite_polynomials(n):
    H = [np.poly1d([1]), np.poly1d([2, 0])]  # H_0(x), H_1(x)
    for k in range(2, n + 1):
        Hk = np.poly1d([2, 0]) * H[-1] - 2 * (k - 1) * H[-2]  # rekurencyjna relacja
        H.append(Hk)
    return H

# Całkowanie metodą Simpsona z wagą e^{-x^2}, wymagane dla ortogonalności Hermite'ów
def simpson_weighted(f, a, b, n):
    if n % 2 == 1:
        n += 1  # liczba podprzedziałów musi być parzysta dla reguły Simpsona
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    w = np.exp(-x**2)  # funkcja wagowa e^{-x^2}
    y = f(x) * w

    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * S

# Oblicza współczynniki a_k rozwinięcia funkcji f w bazie wielomianów Hermite'a
def compute_hermite_coefficients_simpson(f, degree, a, b, n_intervals):
    H_polys = hermite_polynomials(degree)
    coeffs = []
    for i in range(degree + 1):
        Hi = H_polys[i]
        # funkcja podcałkowa: f(x) * H_i(x)
        numerator_func = (lambda H: lambda x: f(x) * H(x))(Hi)
        num = simpson_weighted(numerator_func, a, b, n_intervals)
        # norma L^2 dla H_i z wagą e^{-x^2}
        norm = math.sqrt(math.pi) * (2 ** i) * math.factorial(i)
        coeffs.append(num / norm)
    return coeffs, H_polys

# Oblicza wartości aproksymowanej funkcji na podstawie współczynników i bazowych H_i
def horner_evaluate(coeffs, H_polys, x_vals):
    y_vals = np.zeros_like(x_vals, dtype=float)
    for a, H in zip(coeffs, H_polys):
        y_vals += a * H(x_vals)
    return y_vals

# Interfejs główny programu
def main():
    print("Wybierz funkcję do aproksymacji:")
    for key in functions:
        print(f"{key}: {functions[key][0]}")

    choice = input("Twój wybór: ").strip()
    fname, f_orig = functions.get(choice, functions['1'])

    # Wczytanie parametrów aproksymacji
    a = float(input("Lewy koniec przedziału aproksymacji: "))
    b = float(input("Prawy koniec przedziału aproksymacji: "))
    degree = int(input("Stopień wielomianu Hermite'a: "))

    # Automatyczne ustalenie sensownej liczby przedziałów do całkowania
    n_intervals = max(50, 10 * degree)
    if n_intervals % 2 == 1:
        n_intervals += 1

    f = np.vectorize(f_orig)
    coeffs, H_polys = compute_hermite_coefficients_simpson(f_orig, degree, a, b, n_intervals)

    # Obliczenie wartości aproksymacji i błędu RMS
    x_vals = np.linspace(a, b, 500)
    y_true = f(x_vals)
    y_approx = horner_evaluate(coeffs, H_polys, x_vals)

    error = np.sqrt(np.mean((y_true - y_approx) ** 2))
    print(f"\nRMS Błąd aproksymacji: {error:.5e}")

    # Wizualizacja wyników
    plt.plot(x_vals, y_true, label=f"Oryginalna: {fname}")
    plt.plot(x_vals, y_approx, '--', label=f"Aproksymacja (stopień {degree})")
    plt.title("Aproksymacja wielomianami Hermite'a")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
