# Megerősítéses tanulás kötelező program

A feladat egy [Sokoban](https://hu.wikipedia.org/wiki/Sz%C3%B3koban) ágens készítése. A játék célja, hogy egy négyzehálós játéktéren dobozokat toljunk előre megadott helyekre.
<!-- A játék a Sokoban egy egyszerűsített változata, ahol minden lépésben egy elemet kell ledobnunk. -->

<!-- Egy lépésben két paramétert kell beállítanunk, hogy melyik oszlopba rakjuk le az elemet, és, hogy az elem a 4 forgatási iránya közül melyikbe álljon. -->

# Sokoban

 A sokoban (倉庫番, „raktáros”) egy olyan fejtörő, ahol a játékosnak egy felülnézetes labirintusban kell dobozokat tologatnia a helyükre. Egyszerre csak egy dobozt lehet mozgatni, és csak tolni lehet, húzni nem. [ - Wikipédia](http://sokobano.de/wiki/index.php?title=Main_Page)

A játékot bárki kipróbálhatja például az alábbi oldalon: [www.sokobanonline.com](https://www.sokobanonline.com/play/lessons/2246_lesson-1-1)


<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/4b/Sokoban_ani.gif"><br/>
</p>


# Környezet

A [gymnasium](https://gymnasium.farama.org/index.html) környezetben az alábbi szabályok érvényesek: a cél, hogy az összes doboz valamelyik célba bekerüljön. Ehhez 4 fajta mozgást tud végezni a játékos: fel, le, jobbra és balra tud egyet lépni. Csak üres helyre, célba, vagy olyan helyre lehet lépni, ahol van egy doboz, de az a lépés hatására el tud tolódni. Dobozt csak üres helyre vagy célba lehet betolni, és egyszerre csak egy dobozt lehetséges mozgatni.


|   |   |
|-------------------|------------------------------|
| Action space      | <pre>Discrete(5, start=1)</pre> |
| Observation space | <pre>Box(0, 4, (*paddedSize*))</pre>|


### Akciók

A következő ID-jű akciókat lehet végrehajtani a környezetben:

| ID | Akció  |
|----|--------|
| 1  | Fel    |
| 2  | Le     |
| 3  | Balra  |
| 4  | Jobbra |

### Megfigyelések

A megfigyeléseket egy numpy tömbben kapjuk meg, és a játékbeli elemeket a következő számok jelölik:

| ID | Állapot |
|----|---------|
| 0  | Üres    |
| 1  | Fal     |
| 2  | Doboz   |
| 3  | Cél     |
| 4  | Játékos |

> Megjegyzés: Amennyiben felül kell definiálni az `observation`-t, az `info` dictionary `grid` elemében továbbra is elérhető lesz az eredeti tömb.

### Jutalomfüggvény

Az alapértelmezett jutalomfüggvény minden lépésben ellenőrzi, hogy az összes doboz célba ért-e, ha igen $1$-et, ha nem $0$ jutalmat ad. Ez természetesen egy `RewardWrapper` segítségével felülírható.


### Paraméterek

A `SokobanEnv` létrehozásakor a következő paramétereket tudjuk változtatni:

- **render_mode:** 'text' módban a konzolra kiírja minden lépésben az aktuális *grid*-et. 'rgb_array' módban használható a `RecordVideo` wrapper, alapértelmezetten None.
- **size:** a bejárható játéktér mérete, ebbe beletartozik a pályát körülvevő fal is. Alapértelmezetten (5, 5) tuple. <!-- TODO -->
- **padded_size:** a *size* méretű játékteret falakkal lehet kibővíteni a megadott méretre. Ez akkor lehet hasznos ha például egy (5, 5) méretű játékteren tanított modellt szeretnénk később kiértékelni (7, 7) méretű játéktérrel is. Alapértelmezetten (7, 7) tuple. <!-- TODO -->
- **num_boxes:** a játéktérre kerülő dobozok száma (természetesen ugyanennyi cél is kerül a játéktérre). Alapértelmezetten 2 int, de lehet list(int) is, például [1, 2, 3], ekkor minden egyes pályageneráláskor véletlenszerűen 1, 2, vagy 3 doboz lesz. <!-- TODO -->
- **time_limit:** A maximális lépésszám epizódonként. Alapértelmezetten 50 int.

# Pontszámítás <!-- TODO -->

A pontszámítás megegyezik az `evaluate` fügvény eredményével, azaz: $$\text{Score} = \frac{\text{epizódokban kapott rewardok összege}}{\text{epizódok száma}}$$

# Telepítés és futtatás

A rendszer egyaránt használható Google Colabon és lokálisan is. A környezet egy átlagos laptop processzorán is kényelmesen futtatható.

Példa [Colab notebook](https://colab.research.google.com/drive/1hDlN6tgv2bRXcPK__GNrzTeX0WtrqH5H?usp=sharing).

Az alábbi útmutatóban [conda](https://docs.conda.io/en/latest/) virtuális környezetet fogunk használni.

Conda környezet létrehozása:

```bash
conda create -n sokoban_gym python=3.11
conda activate sokoban_gym
```

Rendszer letöltése és a csomagok telepítése:

```bash
git clone https://github.com/szegedai/sokoban_gym_szte.git

cd sokoban_gym_szte

pip install -r requirements.txt
```

Példakód kipróbálása:

```bash
python example_base.py
```
