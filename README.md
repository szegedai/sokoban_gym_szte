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

Példa [Colab notebook](<!-- TODO -->).

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

# Követelmények

A végleges környezet a hagyományos <!-- TODO --> (10, 10) méretű táblát használ, és maximálisan 4 <!-- TODO --> dobozt kell a helyére pakolni.

Az ágenst a [agent/agent.py](agent/agent.py) fájlban kell megvalósítani. Ezt fogja meghívni a végleges kiértékelő rendszer minden egyes lépésben.

Az ágensben az *act* metódust kell módosítani, ami a környezetből kapott megfigyelés alapján visszaadja a következő lépést.

Ezzen felül a konstruktorban lehetőség van az ágensünk inicializálására, például egy korábban tanult modell betöltésére. Illetve, amennyiben használtunk *wrapper*-eket a környezet modosításához azokat is lehtőségünk van itt létrehozni.

Egy példa ágens, ami egy betanított Stable-Baselines3 modellt használ az alábbi módon nézz ki:
  
  ```python
  from stable_baselines3 import A2C
from sokoban_gym.wrappers.observation import ImageObservationWrapper

class Agent:
    """
    A kötelező programként beadandó ágens leírása.
    """

    def __init__(self, env) -> None:
        """
        A konsztruktorban van lehetőség például a modell betöltésére
        vagy a környezet wrapper-ekkel való kiterjesztésére.
        """
        
        self.model = A2C.load("models/Sokoban-v1_5_8_1box_A2C_CNN")
        
        # A környezetet kiterjeszthetjük wrapper-ek segítségével.
        # Ha tanításkor modosítottuk a megfigyeléseket,
        # akkor azt a módosítást kiértékeléskor is meg kell adnunk.
        self.observation_wrapper = ImageObservationWrapper(env)

    def act(self, observation):
        """
        A megfigyelés alapján visszaadja a következő lépést.
        Ez a függvény fogja megadni az ágens működését.
        """

        # Ha tanításkor modosítottuk a megfigyeléseket,
        # akkor azt a módosítást kiértékeléskor is meg kell adnunk.
        extended_obsetvation = self.observation_wrapper.observation(observation)

        return self.model.predict(extended_obsetvation, deterministic=True)
  ```

## Felhasználható csomagok

Természetesen a Stable-Baselines3 használata nem kötelező, lehetőség van tetszőleges modell, illetve egyénileg írt kód használatára is.

A kiértékelő rendszerben az alábbi csomagok vannak telepítve.
<!-- TODO -->

Új csomagok telepíthetők, ha erre van igényetek kérlek jelezzétek a kötelező programhoz létrehozott coospace forumon.

## Ranglista

A ranglista és a feltöltés az alábbi oldalon érhető el:

[https://chatbot-rgai3.inf.u-szeged.hu/rl/](https://chatbot-rgai3.inf.u-szeged.hu/rl/)

## Feltöltés

Az elkészült kódokat fel kell tötenetek HuggingFace-re. Majd, ha úgyérzitek, hogy minden rendben van, akkor a [ranglista oldalán](https://chatbot-rgai3.inf.u-szeged.hu/rl/upload/) tudjátok elindítani a hivatalos kiértékelést. Ehhez meg kell adnotok a HuggingFace repository nevét, ahova feltöltöttétek a kódotokat és a modelleket, a neptun azonosítótokat és egy megjelenítéshez használni kívánt nevet.

A HuggingFace repository-ba mindent fel kell tölteni, ami szükséges a kód futtatásához. Ez magában foglalja a kódokat és a szükséges modelleket. Az *agent.py*-nak a repository gyökérkönyvtárában kell lennie. Példát erre az alábbi repository-ban találtok: [szterlcourse/tetris_example](https://huggingface.co/szterlcourse/tetris_example/tree/main)

### Példa

Az alábbi [notebook](<!-- TODO -->), illetve a lenti parancsok megmutatják hogyan tudtok betanítani, leellenőrizni és feltöteni a Hugging Face-re egy ágenst.

A modellt betíníthatod a [train.py](train.py) fájl segítségével, ez létre fog hozni egy modellt az *agent* mappában.
```bash
python train.py
```

Fontos, hogyha tanításkor modosítottál a tanuló algoritmuson, a megfigyeléseken... vagy csak egyéni szabályokat szeretnél írni, akkor már a korábban említett [agent/agent.py](agent/agent.py) fájlt kell ehhez módosítanod.

A kész ágenst az [evaluate.py](evaluate.py) fájl segítségével ellenőrizheted.
```bash
python evaluate.py
```

A modellt feltöltheted a HuggingFace-re a [upload.py](upload.py) fájl segítségével.

Ehhez viszont először meg kell adnod a fájlban a létrehozni (vagy felülírni) kívánt repository nevét. Illetve a Hugging Face tokenedet. Ezt az alábbi helyen tudod létrehozni a Hugging Face-en belül: [Settings/Access Tokens](https://huggingface.co/settings/token)

```python
# Ezt át kell írni a saját felhasználónevedre és az általad választott repó nevére
# Pl.: "szterlcourse/my_agent"
repo_id = ""

# Ide be kell írni a saját tokenedet, amit a Hugging Face oldalán tudsz létrehozni (https://huggingface.co/settings/token)
token = ""
```

Ha ezek megvannak, akkor a fájl futtatásával feltöltheted a Hugging Face-re a kódot és a modelleket.
```bash
python evaluate.py
```

A feltöltést kézzel is megteheted, de ezekben van arra példa, hogy hogyan lehet kódból létrehozni egy Hugging Face repository-t és feltölteni bele a kódot és a modelleket.

## Hibák megjelentetése

Az utolsó feltöltés log-ját a neptun kódotok segítségébvel az alábbi link módosításával tudjátok megnézni:

[https://chatbot-rgai3.inf.u-szeged.hu/rl/log/\<NEPTUNKOD\>/](https://chatbot-rgai3.inf.u-szeged.hu/rl/log/NEPTUNKOD/)

Fontos, hogy a záró / szükséges.

# Követelmények

<!-- TODO -->
A kötelező programért szerezhető 30 pont begyűjtéséhez fel kell töltened egy rendszert, ami a szerveren történő kiértékeléskor legalább 40 score-t ér el.

A legjobb 5 felöltő mentesül az elméleti zh alól.

A további helyezések extra pluszpontokat érnek, amiknek a pontos szabályait a későbbiekben részletezzük.
