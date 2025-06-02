#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aproksymacja wielomianami Hermite'a (wariant 2) – wszystko w jednym pliku,
z czterema domyślnymi funkcjami do wyboru.
"""

import math
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

# -------------------------------------------------------------------
# 1. Schemat Hornera
# -------------------------------------------------------------------
def horner(x: float, coeffs: List[float]) -> float:
    """
    Oblicza wartość wielomianu w punkcie x, używając schematu Hornera.
    coeffs: lista współczynników [a_0, a_1, ..., a_n],
    gdzie P(x) = a_0 + a_1*x + ... + a_n*x^n.
    """
    result = 0.0
    for a in reversed(coeffs):
        result = result * x + a
    return result

# -------------------------------------------------------------------
# 2. Wielomiany Hermite’a (fizykalistyczne)
# -------------------------------------------------------------------
def hermite_coeffs(n: int) -> List[float]:
    """
    Zwraca współczynniki H_n(x) (w porządku rosnących potęg x) dla fizykalistycznego
    wielomianu Hermite'a:
      H_0(x) = 1
      H_1(x) = 2x
      H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)
    """
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    H_prev: List[float] = [1.0]       # H_0
    H_curr: List[float] = [0.0, 2.0]  # H_1
    for k in range(1, n):
        deg_curr = len(H_curr) - 1  # = k
        H_next = [0.0] * (deg_curr + 2)
        # 2 * x * H_curr --> przesunięcie współczynników
        for i, a in enumerate(H_curr):
            H_next[i + 1] += 2.0 * a
        # -2k * H_prev (wyrównanie stopni)
        factor = -2.0 * k
        for i, a in enumerate(H_prev):
            H_next[i] += factor * a
        H_prev, H_curr = H_curr, H_next
    return H_curr

def hermite_eval(x: float, coeffs: List[float]) -> float:
    """
    Ocena wielomianu Hermite'a o danej liście współczynników w punkcie x,
    używając schematu Hornera.
    """
    return horner(x, coeffs)

# -------------------------------------------------------------------
# 3. Całkowanie – Simpsona ze współczynnikiem wagowym, adaptacyjne
# -------------------------------------------------------------------
def simpson_weighted_general(f: Callable[[float], float],
                             w: Callable[[float], float],
                             a: float,
                             b: float,
                             n: int) -> float:
    """
    Oblicza ∫_a^b f(x) * w(x) dx metodą Simpsona (kompozytową) przy n podprzedziałach (n parzyste).
    """
    h = (b - a) / n
    result = 0.0
    for i in range(0, n, 2):
        x0 = a + i * h
        x1 = x0 + h
        x2 = x0 + 2 * h
        result += (h / 3) * (f(x0) * w(x0) + 4.0 * f(x1) * w(x1) + f(x2) * w(x2))
    return result

def adaptive_simpson_general(f: Callable[[float], float],
                             w: Callable[[float], float],
                             a: float,
                             b: float,
                             eps: float = 1e-6,
                             initial_n: int = 4,
                             max_n: int = 1 << 16) -> Tuple[float, int]:
    """
    Adaptacyjnie oblicza ∫_a^b f(x) * w(x) dx metodą Simpsona, zwiększając n (parzyste)
    aż do osiągnięcia dokładności eps lub do max_n. Zwraca (wartość całki, użyte n).
    """
    n = initial_n if initial_n % 2 == 0 else initial_n + 1
    prev = simpson_weighted_general(f, w, a, b, n)
    while n * 2 <= max_n:
        n *= 2
        curr = simpson_weighted_general(f, w, a, b, n)
        if abs(curr - prev) < eps:
            return curr, n
        prev = curr
    return prev, n

# -------------------------------------------------------------------
# 4. Aproksymacja – współczynniki c_k oraz ocena wielomianu
# -------------------------------------------------------------------
def compute_coefficients(f: Callable[[float], float],
                         a: float,
                         b: float,
                         N: int,
                         eps: float,
                         initial_n: int) -> Tuple[List[float], List[List[float]]]:
    """
    Oblicza współczynniki c_0..c_N dla aproksymacji f przez wielomiany Hermite'a
    stopnia N na [a,b] z wagą w(x)=exp(-x^2). Zwraca (lista c_k, lista list współczynników H_k).
    """
    w = lambda x: math.exp(-x * x)
    c_list: List[float] = []
    herm_coeffs_list: List[List[float]] = []

    for k in range(N + 1):
        Hk = hermite_coeffs(k)
        herm_coeffs_list.append(Hk)
        f_Hk = lambda x, coeffs=Hk: hermite_eval(x, coeffs)
        numerator, _ = adaptive_simpson_general(
            lambda x: f(x),
            lambda x: f_Hk(x) * w(x),
            a, b,
            eps=eps / 10,
            initial_n=initial_n
        )
        denominator, _ = adaptive_simpson_general(
            lambda x: f_Hk(x),
            lambda x: f_Hk(x) * w(x),
            a, b,
            eps=eps / 10,
            initial_n=initial_n
        )
        if abs(denominator) < 1e-16:
            c_k = 0.0
        else:
            c_k = numerator / denominator
        c_list.append(c_k)

    return c_list, herm_coeffs_list

def evaluate_approximation(x: float,
                           c_list: List[float],
                           herm_coeffs_list: List[List[float]]) -> float:
    """
    Ocena wielomianu aproksymacyjnego p_N(x) = sum_{k=0}^N c_k * H_k(x).
    """
    result = 0.0
    for k, c_k in enumerate(c_list):
        coeffs_k = herm_coeffs_list[k]
        result += c_k * hermite_eval(x, coeffs_k)
    return result

def compute_error_L2(f: Callable[[float], float],
                     p_eval: Callable[[float], float],
                     a: float,
                     b: float,
                     eps: float,
                     initial_n: int) -> float:
    """
    Oblicza błąd L² wagowy: sqrt( ∫_a^b [f(x) - p(x)]^2 * w(x) dx ).
    """
    w = lambda x: math.exp(-x * x)
    integral, _ = adaptive_simpson_general(
        lambda x: f(x) - p_eval(x),
        lambda x: (f(x) - p_eval(x)) * w(x),
        a, b,
        eps=eps / 10,
        initial_n=initial_n
    )
    return math.sqrt(integral)

# -------------------------------------------------------------------
# 5. Narzędzia do rysowania wykresów
# -------------------------------------------------------------------
def plot_function_pair(f: Callable[[float], float],
                       p: Callable[[float], float],
                       a: float,
                       b: float,
                       num_points: int = 500) -> None:
    """
    Rysuje na jednym wykresie funkcję oryginalną f(x) i aproksymację p(x).
    """
    xs = [a + i * (b - a) / (num_points - 1) for i in range(num_points)]
    ys_f = [f(x) for x in xs]
    ys_p = [p(x) for x in xs]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys_f, label="f(x) (oryginał)")
    plt.plot(xs, ys_p, '--', label="p_N(x) (aproksymacja)")
    plt.title("Porównanie: f(x) vs. p_N(x)")
    plt.xlabel("x")
    plt.ylabel("wartość funkcji")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 6. Menu główne i logika programu
# -------------------------------------------------------------------
def choose_function() -> Callable[[float], float]:
    """
    Pozwala użytkownikowi wybrać jedną z czterech domyślnych funkcji do aproksymacji:
      1) f(x) = x
      2) f(x) = |x|
      3) f(x) = x^2 + 2x + 1
      4) f(x) = sin(x)
    """
    print("Wybierz jedną z czterech domyślnych funkcji do aproksymacji:")
    print("1) f(x) = x")
    print("2) f(x) = |x|")
    print("3) f(x) = x^2 + 2x + 1")
    print("4) f(x) = sin(x)")
    choice = input("Twój wybór (1–4): ").strip()

    if choice == '1':
        return lambda x: x
    elif choice == '2':
        return lambda x: abs(x)
    elif choice == '3':
        # współczynniki [1, 2, 1] oznaczają 1 + 2x + x^2
        return lambda x: horner(x, [1.0, 2.0, 1.0])
    else:
        # choice == '4'
        return lambda x: math.sin(x)

def main():
    print("=== Aproksymacja wielomianami Hermite'a (wariant 2) ===")
    f = choose_function()

    a = float(input("Podaj lewą granicę przedziału aproksymacji a: "))
    b = float(input("Podaj prawą granicę przedziału aproksymacji b: "))
    if b <= a:
        print("  Błąd: b musi być większe niż a. Kończę.")
        return

    print("\nTryb działania:")
    print("1) Stały stopień wielomianu aproksymacyjnego")
    print("2) Podajemy docelowy błąd, program dobiera stopień iteracyjnie")
    mode = input("Twój wybór (1–2): ").strip()

    eps_integ = float(input("Podaj ε dla całkowania (np. 1e-6): "))
    initial_n = int(input("Podaj początkową liczbę podprzedziałów (np. 4): "))
    if initial_n % 2 != 0:
        initial_n += 1  # zapewniamy, że jest parzyste

    if mode == '1':
        N = int(input("Podaj stopień wielomianu aproksymacyjnego N: "))
        print("\n=== Obliczanie współczynników ===")
        c_list, herm_coeffs_list = compute_coefficients(f, a, b, N, eps_integ, initial_n)

        def pN_fixed(x: float) -> float:
            return evaluate_approximation(x, c_list, herm_coeffs_list)

        err = compute_error_L2(f, pN_fixed, a, b, eps_integ, initial_n)

        print(f"\nStopień N = {N}")
        print("Współczynniki c_k (k=0..N):")
        for k, c in enumerate(c_list):
            print(f"  c_{k} = {c:.6e}")
        print(f"Błąd L²-ważony: {err:.6e}")
        plot_function_pair(f, pN_fixed, a, b)

    else:
        eps_target = float(input("Podaj docelowy błąd L² (np. 1e-4): "))
        print("\n=== Dobieranie stopnia iteracyjnie aż do eps_target ===")
        N = 0
        found = False

        while True:
            c_list, herm_coeffs_list = compute_coefficients(f, a, b, N, eps_integ, initial_n)

            def pN_iter(x: float, c_list=c_list, herm_coeffs_list=herm_coeffs_list) -> float:
                return evaluate_approximation(x, c_list, herm_coeffs_list)

            err = compute_error_L2(f, pN_iter, a, b, eps_integ, initial_n)
            print(f"  N = {N:2d}  => błąd L² = {err:.3e}")
            if err <= eps_target:
                found = True
                break
            N += 1
            if N > 50:
                print("  Nie udało się osiągnąć zadanego błędu do N=50. Kończę pętlę.")
                break

        if found:
            print("\n=== Wynik końcowy ===")
            print(f"Uzyskano błąd {err:.3e} przy N = {N}")
            print("Współczynniki c_k:")
            for k, c in enumerate(c_list):
                print(f"  c_{k} = {c:.6e}")
            plot_function_pair(f, pN_iter, a, b)

    print("\n=== Koniec programu ===")

if __name__ == "__main__":
    main()
