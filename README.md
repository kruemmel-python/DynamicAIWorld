[![mas](https://ciphercore.de/1mas.svg)](https://ciphercore.de/1mas.svg)


# DynamicAIWorld

DynamicAIWorld ist ein Framework zur Simulation einer lebendigen und dynamischen Umgebung, in der mehrere KI-gesteuerte Agenten miteinander interagieren, Ressourcen sammeln, Ziele verfolgen und Hindernissen ausweichen. Es kombiniert **Deep Reinforcement Learning** mit einer interaktiven Umgebung, die in **Pygame** visualisiert wird.

---

## Features

- **Multi-Agenten-System**: Bis zu 4 Agenten, die gleichzeitig agieren.
- **Ressourcenmanagement**: Agenten können Ressourcen sammeln, die dynamisch neu erscheinen.
- **Bewegliche Hindernisse und Ressourcen**: Eine realistische Simulation durch dynamische Umgebungselemente.
- **Reinforcement Learning mit DDQN**: Nutzen von **Deep Q-Learning** für Entscheidungsfindung und Anpassung.
- **Zielverfolgung**: Agenten erhalten Belohnungen für das Erreichen spezifischer Ziele.
- **Kommunikation zwischen Agenten**: Informationen können zwischen Agenten ausgetauscht werden.
- **Anpassbare Umgebung**: Unterstützung für verschiedene Ressourcentypen, Hindernisdichten und Kommunikationsreichweiten.

---

## Technologien

- **Python**: Programmiersprache.
- **Pygame**: Visualisierung und Interaktion.
- **PyTorch**: KI-Modelltraining und -Implementierung.
- **NumPy**: Effiziente Datenmanipulation.
- **Reinforcement Learning**: Deep Double Q-Learning (DDQN) mit Prioritized Experience Replay.

---

## Installation

1. **Voraussetzungen:**
   - Python 3.8+
   - Installierte Bibliotheken: `pygame`, `torch`, `numpy`

2. **Schritte:**
   ```bash
   # Klonen Sie das Repository
   git clone https://github.com/kruemmel-python/DynamicAIWorld.git
   cd DynamicAIWorld

   # Installieren Sie die Abhängigkeiten
   pip install -r requirements.txt
   ```

3. **Starten der Simulation:**
   ```bash
   python main.py
   ```

---

## Spielanleitung

- **Agentensteuerung**: Die Agenten werden vom KI-Modell gesteuert.
- **Ziel**: Agenten sollen Ressourcen sammeln, Hindernissen ausweichen und ihre Ziele erreichen.
- **Belohnungssystem**:
  - Ressourcen sammeln: +1 Punkt
  - Ziel erreichen: +10 Punkte
  - Hindernisberührung: -1 Punkt
  - Untätigkeit: -0.1 Punkte
  - Energieverlust: -2 Punkte (bei Energieverlust auf 0)

---

## Projektstruktur

```
DynamicAIWorld/
├── main.py          # Hauptdatei, die die Simulation startet
├── requirements.txt # Abhängigkeiten
└── README.md        # Dokumentation
```

---

## Anpassung

Passen Sie die Umgebung über folgende Variablen in der Datei `main.py` an:

- **Größe des Gitters**: `GRID_SIZE`
- **Anzahl der Agenten**: `NUM_AGENTS`
- **Ressourcentypen und Dichte**:
  ```python
  resources = {
      1: 0.1,  # Erz
      2: 0.1,  # Bäume
      3: 0.1,  # Wasser
      4: 0.05, # Pflanzen
      6: 0.05  # Seltene Ressourcen
  }
  ```

- **Kommunikationsreichweite**: `COMMUNICATION_RANGE`
- **Hindernisdichte**: `OBSTACLE_DENSITY`

---

## Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

---

## Autor

Entwickelt von **Ralf Krümmel**.

- [GitHub-Profil](https://github.com/kruemmel-python)
- Bei Fragen oder Vorschlägen können Sie mich gerne kontaktieren.

---

## Ausblick

DynamicAIWorld ist mehr als eine Simulation – es ist ein Grundstein für:

- Städteplanungssimulationen
- Dynamische Verkehrsmanagement-Systeme
- Lern- und Lehrzwecke für Reinforcement Learning
- Forschung im Bereich Multi-Agenten-Systeme

**Lassen Sie Ihrer Kreativität freien Lauf und gestalten Sie Ihre eigene dynamische Welt!**
