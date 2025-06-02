import math
import numpy as np
import matplotlib.pyplot as plt

# Przykładowe funkcje do aproksymacji
def f1(x): return x
def f2(x): return np.abs(x)
def f3(x): return x**3 - 2 * x + 1
def f4(x): return np.sin(x)
def f5(x): return np.cos(x**2)

# Słownik dostępnych funkcji z opisami
funkcje = {
    '1': ("x", f1),
    '2': ("|x|", f2),
    '3': ("x^3 - 2x + 1", f3),
    '4': ("sin(x)", f4),
    '5': ("cos(x^2)", f5)
}

# Generuje wielomiany Hermite'a do stopnia n włącznie
def wielomiany_hermitea(n):
    H = [np.poly1d([1]), np.poly1d([2, 0])]  # H_0(x), H_1(x)
    for k in range(2, n + 1):
        Hk = np.poly1d([2, 0]) * H[-1] - 2 * (k - 1) * H[-2]  # rekurencyjna relacja
        H.append(Hk)
    return H

# Całkowanie metodą Simpsona z wagą e^{-x^2}, wymagane dla ortogonalności Hermite'ów
def simpson_z_waga(f, a, b, n):
    if n % 2 == 1:
        n += 1  # liczba podprzedziałów musi być parzysta dla reguły Simpsona
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    w = np.exp(-x**2)  # funkcja wagowa e^{-x^2}
    y = f(x) * w

    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * S

# Oblicza współczynniki a_k rozwinięcia funkcji f w bazie wielomianów Hermite'a
def oblicz_wspolczynniki_hermitea_simpson(f, stopien, a, b, liczba_przedzialow):
    wielomiany_H = wielomiany_hermitea(stopien)
    wspolczynniki = []
    for i in range(stopien + 1):
        Hi = wielomiany_H[i]
        # funkcja podcałkowa: f(x) * H_i(x)
        funkcja_licznika = (lambda H: lambda x: f(x) * H(x))(Hi)
        licznik = simpson_z_waga(funkcja_licznika, a, b, liczba_przedzialow)
        # norma L^2 dla H_i z wagą e^{-x^2}
        norma = math.sqrt(math.pi) * (2 ** i) * math.factorial(i)
        wspolczynniki.append(licznik / norma)
    return wspolczynniki, wielomiany_H

# Oblicza wartości aproksymowanej funkcji na podstawie współczynników i bazowych H_i
def oblicz_wartosci_hornera(wspolczynniki, wielomiany_H, wartosci_x):
    wartosci_y = np.zeros_like(wartosci_x, dtype=float)
    for a, H in zip(wspolczynniki, wielomiany_H):
        wartosci_y += a * H(wartosci_x)
    return wartosci_y

def main():
    print("Wybierz funkcję do aproksymacji:")
    for klucz in funkcje:
        print(f"{klucz}: {funkcje[klucz][0]}")

    wybor = input("Twój wybór: ").strip()
    nazwa_funkcji, funkcja_oryg = funkcje.get(wybor, funkcje['1'])

    # Wczytanie parametrów aproksymacji
    a = float(input("Lewy koniec przedziału aproksymacji: "))
    b = float(input("Prawy koniec przedziału aproksymacji: "))
    stopien = int(input("Stopień wielomianu Hermite'a: "))

    # Automatyczne ustalenie sensownej liczby przedziałów do całkowania
    liczba_przedzialow = max(100, 20 * stopien)
    if liczba_przedzialow % 2 == 1:
        liczba_przedzialow += 1

    f = np.vectorize(funkcja_oryg)
    wspolczynniki, wielomiany_H = oblicz_wspolczynniki_hermitea_simpson(funkcja_oryg, stopien, a, b, liczba_przedzialow)

    # Obliczenie wartości aproksymacji i błędu RMS
    wartosci_x = np.linspace(a, b, 500)
    y_prawdziwe = f(wartosci_x)
    y_aproksymowane = oblicz_wartosci_hornera(wspolczynniki, wielomiany_H, wartosci_x)

    blad = np.sqrt(np.mean((y_prawdziwe - y_aproksymowane) ** 2))
    print(f"\nRMS Błąd aproksymacji: {blad:.5e}")

    # Wizualizacja wyników
    plt.plot(wartosci_x, y_prawdziwe, label=f"Oryginalna: {nazwa_funkcji}")
    plt.plot(wartosci_x, y_aproksymowane, '--', label=f"Aproksymacja (stopień {stopien})")
    plt.title("Aproksymacja wielomianami Hermite'a")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()