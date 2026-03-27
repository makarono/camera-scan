# Skener SD kartica nadzornih kamera

Skripta automatski skenira SD kartice iz nadzornih kamera i detektira snimke na kojima se nalaze ljudi, životinje ili vozila. Koristi AI detekciju objekata kako bi filtrirala lažne okidače (kiša, vjetar, drveće).

## Zahtjevi

### Lokalno pokretanje
- Python 3
- [uv](https://github.com/astral-sh/uv) - za upravljanje virtualnim okruženjem
- [just](https://github.com/casey/just) - za pokretanje zadataka
- [VLC](https://www.videolan.org/) - za pregled rezultata

### Docker
- Docker i Docker Compose

## Pokretanje

### Lokalno

```bash
# Puni scan (YOLOv8s + YOLO-World) - default
just

# Brzi scan (samo YOLOv8s)
just fast

# Samo instalacija virtualnog okruženja
just setup

# Brisanje rezultata
just clean
```

Pri prvom pokretanju automatski se kreira virtualno okruženje i instaliraju ovisnosti.

### Docker

```bash
# Build Docker image
just docker-build

# Puni scan u Dockeru
just docker-scan

# Brzi scan u Dockeru
just docker-fast
```

## Mountanje SD kartice u Docker

### macOS

1. Umetnuti SD karticu - automatski se mountira u `/Volumes/`
2. Provjeriti naziv: `ls /Volumes/`
3. Urediti `docker-compose.yml` - prilagoditi volume putanju:

```yaml
volumes:
  - ./results:/app/results
  - /Volumes/NAZIV_KARTICE:/sdcard:ro
```

Primjer za karticu "NO NAME":
```yaml
  - /Volumes/NO NAME:/sdcard:ro
```

### Linux

1. Umetnuti SD karticu
2. Provjeriti mount point: `lsblk` ili `df -h`
3. Urediti `docker-compose.yml`:

```yaml
volumes:
  - ./results:/app/results
  - /media/$USER/NAZIV_KARTICE:/sdcard:ro
```

Ili ako je ručno mountana:
```yaml
  - /mnt/sdcard:/sdcard:ro
```

### Ručno mountanje na Linuxu

```bash
# Pronađi uređaj
lsblk

# Mountaj (npr. /dev/sdb1)
sudo mkdir -p /mnt/sdcard
sudo mount -o ro /dev/sdb1 /mnt/sdcard

# Nakon pregleda
sudo umount /mnt/sdcard
```

### Napomene

- `:ro` na kraju volume mape znači read-only - kartica se neće mijenjati
- Rezultati se spremaju u `./results/` na hostu putem volume mape
- VLC se ne otvara u Docker modu - playlista se generira u `results/` folderu, otvoriti ručno
- Ako se naziv kartice mijenja, treba urediti `docker-compose.yml` prije svakog pokretanja
- Za brzu promjenu putanje bez uređivanja compose fajla:

```bash
docker compose run --rm -v "/Volumes/NOVA KARTICA:/sdcard:ro" scanner
```

## Kako radi

### Dva moda rada

| Mod | Lokalno | Docker | Modeli | Detekcija |
|-----|---------|--------|--------|-----------|
| Puni | `just scan` | `just docker-scan` | YOLOv8s + YOLO-World | Traktor, pickup, divljač |
| Brzi | `just fast` | `just docker-fast` | Samo YOLOv8s | Standardne klase |

### Proces skeniranja

1. Automatski detektira SD karticu (traži DCIM folder)
2. Pita za naziv kamere (npr. front, stala, kapija)
3. **Prolaz 1 (YOLOv8s):** skenira sve slike i videe za standardne objekte
4. **Prolaz 2 (YOLO-World):** skenira samo preskočene fajlove za dodatne klase (traktor, pickup, divljač)
5. Generira izvještaj i VLC playlistu
6. Otvara VLC sa pozitivnim snimkama (lokalni mod)
7. Pita za sljedeću karticu

### Detektirani objekti

**Prolaz 1 (YOLOv8s):** osoba, auto, kamion, motocikl, bicikl, autobus, pas, mačka, konj, krava, ovca, medvjed

**Prolaz 2 (YOLO-World):** traktor, pickup, jelen, divlja svinja, lisica + sve iz prolaza 1

### Optimizacije brzine

- Slike: ~0.5s po slici
- Video: uzorkuje 1 frame/sec, prestaje čim nađe nešto
- Ako slika ima detekciju, preskače pripadajući video (jer su par)
- YOLO-World radi samo na fajlovima koje prvi prolaz preskočio

## Rezultati

Spremaju se u `results/<naziv_kamere>/<datum_vrijeme>/`:

```
results/
  stala/
    2026-03-27_14-30/
      detected.txt    # izvještaj s detekcijama
      detected.m3u    # VLC playlista
  kapija/
    2026-03-27_14-45/
      detected.txt
      detected.m3u
```

VLC playlista referencira fajlove na SD kartici (ili `/sdcard` u Dockeru) - ne vaditi karticu dok se pregledava.

## Struktura projekta

```
filter_camera.py     # glavna skripta
justfile             # zadaci za pokretanje
requirements.txt     # Python ovisnosti
Dockerfile           # Docker image
docker-compose.yml   # Docker Compose konfiguracija
results/             # rezultati skeniranja
```
