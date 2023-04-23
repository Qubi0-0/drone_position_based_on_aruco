traktujmy tego changeloga dodatkowo jako nwm jakąś listę rzeczy do zrobienia czy coś

[23-04-2023]
Poprawiłem samo oznaczanie markerów i przypisywanie im ID, jest też jakiś prototyp macierzy z parametrami kamery. 
Ja bym teraz skupił się na samym znajdowaniu pozycji, jest taka funkcja jak solvePnP(), która w teorii powinna właśnie to robić.
Sugerowałbym dopisanie jakiejś funkcji (np. get3dPoints()) do wyciągania macierzy X, Y, Z z tego pliku CSV dla konkretnego ID w pętli. Do macierzy kamery użyłem numpy, 
te wartości wyciągane z CSV też byłoby fajnie jakby były ładowane do np.array (powinno to działać z opencv).
