def spike_counts(array):
    conteggio = 0
    for i in range(1, len(array)):
        if array[i] < 0 and array[i - 1] > 0:
            conteggio += 1
    return conteggio

# Esempio di utilizzo:
array_di_numeri = [1, -2, 3, -4, 5, -6, 7]
risultato = conta_negativi_preceduti_positivi(array_di_numeri)
print("Il numero di volte in cui un numero negativo è preceduto da uno positivo è:", risultato)
