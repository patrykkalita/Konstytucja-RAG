## Instrukcja obsługi
- Pobieramy folder z plikami projektu
  - W pliku .env wprowadzamy swój klucz api dla OpenAi
- W terminalu, przechodzimy do lokalizacji w której mamy folder z projektem
- Za pomocą poniższej komendy budujemy obraz Docker
```console
docker build -t konstytucja .
```
- Następnie uruchamiamy zbudowany obraz
```console
docker run -p 8501:8501 konstytucja
```
- W przeglądarce uruchamiamy aplikację pod zdefiniowanym wcześniej portem
- Z listy rozwijanej znajdującej się po lewej stronie wybieramy plik konstytucji
- Gdy plik się załaduje możemy wysyłać wiadomości do chatu
