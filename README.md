# MSI_project

# 1. Założenia Ogólne i Struktura Projektu
Cała grupa ma za zadanie zbudować od zera platformę symulacyjną 2D do walk czołgów oraz stworzyć dla niej inteligentne agenty (AI).
Projekt realizowany jest w dwóch frakcjach o różnych zadaniach i kryteriach oceny:
Frakcja Silnika (The Engine Core)
•	Kto: Część zespołów (zgłoszonych lub wybranych).
•	Rola: "Dostawca" Platformy.
•	Zadanie: Zaprojektować architekturę, zdefiniować API, zaimplementować (przy pomocy LLM), zintegrować i przetestować jeden, wspólny silnik symulacyjny.
•	Produkt: Stabilna, działająca aplikacja, który potrafi dynamicznie wczytać dwa moduły sterujące i przeprowadzić bitwę.
•	Kryterium sukcesu: Dostarczenie na czas stabilnego silnika, który pozwoli na rozegranie turnieju.
Frakcja Agentów (The AI Competitors)
•	Kto: Pozostałe zespoły.
•	Rola: "Klient" Platformy.
•	Zadanie: Zaimplementować "mózg" czołgu (IAgentController) w Pythonie, zgodny z API dostarczonym przez Frakcję Silnika.
•	Produkt: Główny plik .py (moduł), opcjonalnie inne komponenty
•	Kryterium sukcesu: Stworzenie jak najinteligentniejszego agenta (wykorzystującego FLC, A*, GA/PSO) i osiągnięcie jak najlepszego wyniku w finałowym turnieju.

# 2. Harmonogram i Fazy Projektu
## Faza 1: Konwencja Konstytucyjna (Tygodnie 1-2)
•	Cel: Ustalenie i "zamrożenie" ostatecznego kontraktu API (w formie klas abstrakcyjnych Pythona).
•	Uczestnicy: WSZYSTKIE zespoły.
•	Proces:
1.	Frakcja Agentów (Klienci) przedstawia swoje wymagania: "Muszę znać swoje HP", "Potrzebuję listy widocznych wrogów", "Jakie akcje mogę wykonać?".
2.	Frakcja Silnika (Dostawcy) negocjuje i proponuje architekturę: "Listę wrogów dostaniecie w stożku widzenia 30°", "Akcje to przyspieszenie, skręt, strzał".
•	Wynik: Jeden, wspólny plik final_api.py, którego nie wolno już zmieniać.

## Faza 2: Rozwój Równoległy (Tygodnie 3-8)

•	Zadanie Frakcji Silnika:
1.	Podział pracy (np. Moduł Fizyki, Moduł Renderowania, Moduł Kolizji, Moduł Ładowania Agentów).
2.	Formułowanie promptów dla agentów kodujących (LLM) w celu wygenerowania kodu bazowego.
3.	INTEGRACJA wygenerowanego kodu. Debugowanie błędów integracyjnych.
4.	Stworzenie "Bota-Manekina" (Dummy AI), który implementuje final_api.py (np. tylko jeździ w kółko) – niezbędny do testowania silnika.

•	Zadanie Frakcji Agentów:
1.	Nie czekacie na silnik! Bierzecie final_api.py.
2.	Tworzycie własne, minimalistyczne środowisko testowe ("mock engine" / "piaskownica").
3.	W tej piaskownicy implementujecie swoje algorytmy AI, na przykład:
•	Nawigację A* (np. na prostej, tekstowej mapie).
•	Sterownik FLC (testujecie, wysyłając fałszywe dane SensorData).
•	Przeprowadzacie trening offline: używacie GA/PSO do optymalizacji parametrów FLC w walce z prostym botem (np. "jedź do przodu") w Waszej piaskownicy.

## Faza 3: Integracja Techniczna – "Dzień Zero" (Tydzień 9)
•	Cel: Pierwsze połączenie Agentów z Silnikiem.
•	Frakcja Silnika publikuje engine_v1.0_beta.
•	Frakcje Agentów próbują uruchomić swoje AI na prawdziwym silniku.
•	Identyfikacja i zgłaszanie krytycznych błędów (crashe, błędy API). Frakcja Silnika ma tydzień na wydanie stabilnej wersji engine_v1.0_release.

## Faza 4: "Próby Wojenne" – Sparingi i Trening (Tygodnie 10-12)
•	Cel: Testowanie, modyfikacja i ewolucyjny trening Waszych AI.

•	Proces:
1.	Frakcja Silnika zapewnia stabilną wersję silnika.
2.	Wszystkie Frakcje Agentów publikują swoje AI_beta.py we wspólnym repozytorium.
3.	Od teraz każdy ma dostęp do silnika i agentów konkurencji.

•	Wasze Zadanie (Frakcja Agentów):
1.	Modyfikacja Ręczna: Uruchamiacie sparingi (np. MojeAI vs AI_Konkurencji). Obserwujecie, wyciągacie wnioski i ręcznie poprawiacie błędy w logice lub regułach FLC.
2.	Trening Automatyczny (Kluczowe!): Używacie Algorytmów Ewolucyjnych (GA/PSO) do automatycznego ulepszania agenta. Wasza funkcja fitness (oceny) nie bazuje już na "Bocie-Manekinie", ale na wynikach sparingów z prawdziwymi konkurentami.

•	Przykład: Wasz skrypt GA w pętli generuje nowe parametry FLC, uruchamia 10 bitew (w trybie "headless") przeciwko innym agentom i oblicza średni wynik, który staje się oceną "fitness" danego osobnika.
•	Deadline: Na koniec Tygodnia 12, każda Frakcja Agentów musi złożyć finalną, "zamrożoną" wersję swojego AI. Żadne zmiany po tym terminie nie są dozwolone.

## Faza 5: Finałowy Turniej (Ostatnie Zajęcia)
•	Prezentujecie swoje rozwiazania.
•	Oficjalny, oceniany turniej "każdy z każdym" (lub system pucharowy).
•	Frakcja Silnika uruchamia engine_v1.0_release na rzutniku.
•	Każda grupa dostarcza pliki z agentami.
•	Obserwujemy efekty i wyłaniamy zwycięzcę.


# 3. Stos Technologiczny i Wymagania
•	Język: Python. (BE, FE dowolnie)
•	Biblioteki Silnika: Frakcja Silnika decyduje o bibliotekach graficznych (np. Pygame, Arcade) i naukowych (np. NumPy). Wszystkie wymagania muszą być jasno określone w pliku requirements.txt.
•	Biblioteki Agentów: Frakcja Agentów może używać standardowych bibliotek (np. numpy, scipy), ale musi zaimplementować logikę kluczowych algorytmów (FLC, A*, GA/PSO) samodzielnie (nie używamy gotowych bibliotek AI typu fuzzycmeans czy deap, chyba że po konsultacji).
•	Kontrola Wersji: Użycie Git jest obowiązkowe dla obu frakcji.

# 4. Produkty Końcowe (Deliverables)
Każdy zespół składa jeden wspólny pakiet.
## Frakcja Silnika
1.	Kod Źródłowy Silnika: Kompletne, działające repozytorium Git (wraz z requirements.txt).
2.	Dokumentacja API: Czysty, skomentowany plik final_api.py.
3.	Raport Architektoniczny (max 5 stron): Opis architektury, podziału modułów, wyzwań integracyjnych oraz procesu pracy z agentami LLM (jakie prompty, jakie problemy?).

## Frakcja Agentów
1.	Plik Agenta AI: Finalna, "zamrożona" wersja (AI_Grupa_X_FINAL.py).
2.	Kod Źródłowy Piaskownicy: Wasz "mock engine" (mock_engine.py) używany w Fazie 2.
3.	Raport AI (max 8 stron): Musi zawierać:
•	Opis architektury agenta.
•	Opis implementacji metod sztucznej inteligencji
•	Kluczowy rozdział: Opis procesu treningu ewolucyjnego (Faza 4). Jak skonstruowaliście funkcję fitness? Jakie algorytmy (GA/PSO) użyliście? Jakie wyniki (np. wykresy) uzyskaliście w sparingach?
