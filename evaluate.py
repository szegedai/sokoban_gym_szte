# Ha az ágenst szeretnéd módosítani, pl másik algoritmust akarsz használni, másik könyvtára, 
# saját megfigyeléseket... akkor a /content/agent/agent.py fájlt kell módosítanod.

# ⚠⚠⚠ Vigyázz, colab-ban a fájlok módosítasai el fognak vesztni, így ha itt akarod szerkeszteni
# nem árt minden lépés után egy biztonsági mentést készíteni róla ⚠⚠⚠
from utils.eval_utils import evaluate_agent_competition

# Az evaluate_agent_competition függvény végzi a modell kiértékelést. A függvény a githubon
# feltüntetett táblázatnak megfelelően 400 feladatot futtat le. A maximálisan lefutatott 
# feladatok számát az `max_task_num` paraméter segítségével állíthatjátok.
# A feladatok száma csak 40 többszöröse lehet, így ha nem 40-el osztható értéket kap
# a függvény, akkor a feladatok száma a 40 azon legnagyobb többszöröse lesz,
# ami még kisebb a paraméterül kapott értéknél.
evaluate_agent_competition(max_task_num=120)