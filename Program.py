import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math


# Funkcje aproksymowane
def funkcja_liniowa(x):
    return 2 * x + 1


def funkcja_abs(x):
    return np.abs(x)


def funkcja_wielomian(x):
    return x ** 3 - 2 * x ** 2 + x + 1


def funkcja_tryg(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)


def funkcja_zlozona(x):
    return np.exp(-x ** 2) * np.sin(x)


# Słownik funkcji
FUNKCJE = {
    1: ("Liniowa: 2x + 1", funkcja_liniowa),
    2: ("Wartość bezwzględna: |x|", funkcja_abs),
    3: ("Wielomian: x³ - 2x² + x + 1", funkcja_wielomian),
    4: ("Trygonometryczna: sin(x) + 0.5cos(2x)", funkcja_tryg),
    5: ("Złożona: e^(-x²)sin(x)", funkcja_zlozona)
}


# Wielomiany Hermite'a - generowanie iteracyjne
def generuj_wielomiany_hermite(stopien):
    """Generuje współczynniki wielomianów Hermite'a do stopnia n włącznie"""
    wielomiany = []

    # H₀(x) = 1
    wielomiany.append([1])

    if stopien >= 1:
        # H₁(x) = 2x
        wielomiany.append([0, 2])

    # Rekurencja: Hₖ(x) = 2x·Hₖ₋₁(x) - 2k·Hₖ₋₂(x)
    for k in range(2, stopien + 1):
        # Pomnażanie przez 2x (przesunięcie i pomnożenie przez 2)
        h_k_minus_1 = wielomiany[k - 1]
        term1 = [0] + [2 * coeff for coeff in h_k_minus_1]

        # Pomnażanie przez -2k
        h_k_minus_2 = wielomiany[k - 2]
        term2 = [-2 * k * coeff for coeff in h_k_minus_2]

        # Dodawanie wielomianów (wyrównanie długości)
        max_len = max(len(term1), len(term2))
        term1.extend([0] * (max_len - len(term1)))
        term2.extend([0] * (max_len - len(term2)))

        h_k = [term1[i] + term2[i] for i in range(max_len)]
        wielomiany.append(h_k)

    return wielomiany


def oblicz_wartosc_wielomianu_horner(wspolczynniki, x):
    """Oblicza wartość wielomianu metodą Hornera"""
    if not wspolczynniki:
        return 0

    wynik = wspolczynniki[-1]
    for i in range(len(wspolczynniki) - 2, -1, -1):
        wynik = wynik * x + wspolczynniki[i]

    return wynik


def oblicz_hermite(n, x, wielomiany_wspolczynniki):
    """Oblicza wartość n-tego wielomianu Hermite'a w punkcie x"""
    if n >= len(wielomiany_wspolczynniki):
        return 0
    return oblicz_wartosc_wielomianu_horner(wielomiany_wspolczynniki[n], x)


# Całkowanie numeryczne - metoda Simpsona adaptowana
def simpson_adaptive(f, a, b, tol=1e-6, max_iter=1000):
    """Adaptacyjna metoda Simpsona dla całkowania"""

    def simpson_basic(f, a, b):
        h = (b - a) / 2
        return h / 3 * (f(a) + 4 * f(a + h) + f(b))

    def simpson_recursive(f, a, b, tol, whole, m):
        if m <= 0:
            return whole

        c = (a + b) / 2
        left = simpson_basic(f, a, c)
        right = simpson_basic(f, c, b)

        if abs(left + right - whole) <= 15 * tol:
            return left + right + (left + right - whole) / 15

        return (simpson_recursive(f, a, c, tol / 2, left, m - 1) +
                simpson_recursive(f, c, b, tol / 2, right, m - 1))

    whole = simpson_basic(f, a, b)
    return simpson_recursive(f, a, b, tol, whole, max_iter)


def waga_hermite(x):
    """Funkcja wagowa dla wielomianów Hermite'a: e^(-x²)"""
    return np.exp(-x ** 2)


def oblicz_wspolczynnik_aproksymacji(f, n, a, b, wielomiany_wspolczynniki):
    """Oblicza współczynnik cₙ dla aproksymacji"""

    def licznik(x):
        return waga_hermite(x) * f(x) * oblicz_hermite(n, x, wielomiany_wspolczynniki)

    def mianownik(x):
        h_n = oblicz_hermite(n, x, wielomiany_wspolczynniki)
        return waga_hermite(x) * h_n * h_n

    # Obliczanie całek numerycznie
    try:
        integral_licznik = simpson_adaptive(licznik, a, b)
        integral_mianownik = simpson_adaptive(mianownik, a, b)

        if abs(integral_mianownik) < 1e-12:
            return 0

        return integral_licznik / integral_mianownik
    except:
        # Fallback do scipy quad w przypadku problemów
        integral_licznik, _ = quad(licznik, a, b)
        integral_mianownik, _ = quad(mianownik, a, b)

        if abs(integral_mianownik) < 1e-12:
            return 0

        return integral_licznik / integral_mianownik


def aproksymacja_hermite(f, stopien, a, b):
    """Główna funkcja aproksymacji wielomianami Hermite'a"""
    wielomiany_wspolczynniki = generuj_wielomiany_hermite(stopien)
    wspolczynniki_aproks = []

    print(f"Obliczanie współczynników aproksymacji...")
    for n in range(stopien + 1):
        c_n = oblicz_wspolczynnik_aproksymacji(f, n, a, b, wielomiany_wspolczynniki)
        wspolczynniki_aproks.append(c_n)
        print(f"c_{n} = {c_n:.6f}")

    def funkcja_aproksymujaca(x):
        wynik = 0
        for n in range(stopien + 1):
            if abs(wspolczynniki_aproks[n]) > 1e-12:
                wynik += wspolczynniki_aproks[n] * oblicz_hermite(n, x, wielomiany_wspolczynniki)
        return wynik

    return funkcja_aproksymujaca, wspolczynniki_aproks, wielomiany_wspolczynniki


def oblicz_blad_aproksymacji(f, f_aproks, a, b, n_punktow=1000):
    """Oblicza błąd aproksymacji (średniokwadratowy)"""
    x_vals = np.linspace(a, b, n_punktow)
    bledy = []

    for x in x_vals:
        try:
            blad = (f(x) - f_aproks(x)) ** 2
            bledy.append(blad)
        except:
            bledy.append(0)

    return np.sqrt(np.mean(bledy))


def dobierz_stopien_automatycznie(f, a, b, docelowy_blad, max_stopien=10):
    """Automatyczny dobór stopnia wielomianu dla osiągnięcia zadanego błędu"""
    print(f"\nAutomatyczny dobór stopnia wielomianu dla błędu < {docelowy_blad}")
    print("=" * 60)

    for stopien in range(1, max_stopien + 1):
        print(f"\nTestowanie stopnia {stopien}...")

        try:
            f_aproks, _, _ = aproksymacja_hermite(f, stopien, a, b)
            blad = oblicz_blad_aproksymacji(f, f_aproks, a, b)

            print(f"Stopień {stopien}: błąd = {blad:.6f}")

            if blad < docelowy_blad:
                print(f"\n✓ Znaleziono odpowiedni stopień: {stopien}")
                return stopien, f_aproks, blad

        except Exception as e:
            print(f"Błąd dla stopnia {stopien}: {e}")
            continue

    print(f"\n⚠ Nie udało się osiągnąć błędu < {docelowy_blad} do stopnia {max_stopien}")
    return max_stopien, None, float('inf')


def rysuj_wykresy(f, f_aproks, a, b, tytul, stopien):
    """Rysuje wykresy funkcji oryginalnej i aproksymującej"""
    x_vals = np.linspace(a, b, 1000)
    y_orig = [f(x) for x in x_vals]
    y_aproks = [f_aproks(x) for x in x_vals]

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_orig, 'b-', linewidth=2, label='Funkcja oryginalna')
    plt.plot(x_vals, y_aproks, 'r--', linewidth=2, label=f'Aproksymacja Hermite (stopień {stopien})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Aproksymacja wielomianami Hermite\'a\n{tytul}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analiza_stopnia(f, nazwa_funkcji, a, b, max_stopien=8):
    """Analiza wpływu stopnia wielomianu na dokładność"""
    print(f"\n{'=' * 60}")
    print(f"ANALIZA WPŁYWU STOPNIA WIELOMIANU")
    print(f"Funkcja: {nazwa_funkcji}")
    print(f"Przedział: [{a}, {b}]")
    print(f"{'=' * 60}")

    stopnie = []
    bledy = []

    for stopien in range(1, max_stopien + 1):
        try:
            print(f"\nStopień {stopien}:")
            f_aproks, wspolczynniki, _ = aproksymacja_hermite(f, stopien, a, b)
            blad = oblicz_blad_aproksymacji(f, f_aproks, a, b)

            stopnie.append(stopien)
            bledy.append(blad)

            print(f"Błąd średniokwadratowy: {blad:.8f}")

        except Exception as e:
            print(f"Błąd obliczeniowy: {e}")

    # Wykres analizy
    if stopnie and bledy:
        plt.figure(figsize=(10, 6))
        plt.semilogy(stopnie, bledy, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Stopień wielomianu')
        plt.ylabel('Błąd średniokwadratowy (skala log)')
        plt.title(f'Wpływ stopnia wielomianu na dokładność aproksymacji\n{nazwa_funkcji}')
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    print("PROGRAM APROKSYMACJI WIELOMIANAMI HERMITE'A")
    print("=" * 50)

    # Wybór funkcji
    print("\nDostępne funkcje:")
    for key, (nazwa, _) in FUNKCJE.items():
        print(f"{key}. {nazwa}")

    while True:
        try:
            wybor_funkcji = int(input("\nWybierz funkcję (1-5): "))
            if wybor_funkcji in FUNKCJE:
                nazwa_funkcji, funkcja = FUNKCJE[wybor_funkcji]
                break
            else:
                print("Nieprawidłowy wybór!")
        except ValueError:
            print("Wprowadź liczbę!")

    # Parametry aproksymacji
    print(f"\nWybrana funkcja: {nazwa_funkcji}")

    try:
        a = float(input("Podaj początek przedziału aproksymacji: "))
        b = float(input("Podaj koniec przedziału aproksymacji: "))

        if a >= b:
            print("Błąd: początek przedziału musi być mniejszy od końca!")
            return

    except ValueError:
        print("Błąd: wprowadź poprawne liczby!")
        return

    # Tryb pracy
    print("\nTryby pracy:")
    print("1. Zadany stopień wielomianu")
    print("2. Automatyczny dobór stopnia (dla oceny 5)")
    print("3. Analiza wpływu stopnia wielomianu")

    try:
        tryb = int(input("Wybierz tryb (1-3): "))
    except ValueError:
        print("Błąd: wprowadź liczbę!")
        return

    if tryb == 1:
        # Tryb ze stałym stopniem
        try:
            stopien = int(input("Podaj stopień wielomianu aproksymacyjnego: "))
            if stopien < 0:
                print("Stopień musi być nieujemny!")
                return
        except ValueError:
            print("Błąd: wprowadź liczbę całkowitą!")
            return

        print(f"\nRozpoczynanie aproksymacji stopnia {stopien}...")
        try:
            f_aproks, wspolczynniki, wielomiany_wsp = aproksymacja_hermite(funkcja, stopien, a, b)

            blad = oblicz_blad_aproksymacji(funkcja, f_aproks, a, b)
            print(f"\nBłąd średniokwadratowy aproksymacji: {blad:.8f}")

            # Wyświetl kilka wartości wielomianów Hermite'a
            print(f"\nPierwsze wielomiany Hermite'a (współczynniki):")
            for i, wsp in enumerate(wielomiany_wsp[:min(6, len(wielomiany_wsp))]):
                print(f"H_{i}(x): {wsp}")

            rysuj_wykresy(funkcja, f_aproks, a, b, nazwa_funkcji, stopien)

        except Exception as e:
            print(f"Błąd podczas aproksymacji: {e}")

    elif tryb == 2:
        # Tryb automatycznego doboru stopnia
        try:
            docelowy_blad = float(input("Podaj oczekiwany błąd aproksymacji: "))
            if docelowy_blad <= 0:
                print("Błąd musi być dodatni!")
                return
        except ValueError:
            print("Błąd: wprowadź poprawną liczbę!")
            return

        stopien, f_aproks, blad = dobierz_stopien_automatycznie(funkcja, a, b, docelowy_blad)

        if f_aproks is not None:
            rysuj_wykresy(funkcja, f_aproks, a, b, nazwa_funkcji, stopien)

    elif tryb == 3:
        # Analiza wpływu stopnia
        try:
            max_stopien = int(input("Podaj maksymalny stopień do analizy (domyślnie 8): ") or "8")
        except ValueError:
            max_stopien = 8

        analiza_stopnia(funkcja, nazwa_funkcji, a, b, max_stopien)

    else:
        print("Nieprawidłowy tryb!")


if __name__ == "__main__":
    main()