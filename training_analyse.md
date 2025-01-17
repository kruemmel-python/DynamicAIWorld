# Fortschrittsbericht: Training eines Multi-Agenten-Modells

## Einleitung

Dieser Bericht dokumentiert den Fortschritt des Trainings unseres Multi-Agenten-Modells. Wir analysieren die Entwicklung der Gesamtbelohnungen, durchschnittlichen Verluste und Lernraten für jeden Agenten über mehrere Episoden. Ziel ist es, einen umfassenden Überblick über die Leistung und Lernfortschritte jedes Agenten zu geben und allgemeine Trends zu identifizieren.

## Gesamtbelohnung pro Episode

### Episodenfortschritt

Wir betrachten die Entwicklung der Gesamtbelohnungen für alle vier Agenten (Agent 0 bis Agent 3) zu den folgenden Episodenzeitpunkten: 20, 40, 60, 80, 100 und 120.

#### Agent 0

- Die Gesamtbelohnungen pro Episode zeigen einen klaren **aufsteigenden Trend**, was auf eine erfolgreiche Lernkurve und eine zunehmende Effektivität des Agenten hinweist.

#### Agent 1

- Ähnlich wie Agent 0 zeigt auch Agent 1 einen **allgemeinen Anstieg** der Gesamtbelohnungen. Dies deutet darauf hin, dass der Agent in seiner Aufgabe ebenfalls erfolgreich lernt.

#### Agent 2

- Die Belohnungen von Agent 2 **steigen ebenfalls kontinuierlich** über die Episoden hinweg. Dies deutet auf eine effektive Leistungssteigerung und Anpassung des Agenten an die Umgebung hin.

#### Agent 3

- Der Agent 3 zeigt eine **stetige Zunahme** der Belohnungen, auch wenn seine Anfangsleistung unter der der anderen Agenten lag. Seine Lernkurve hat sich jedoch im Laufe der Episoden verbessert.

## Durchschnittlicher Verlust pro Episode

### Episodenfortschritt

Die folgenden Analysen zeigen die Entwicklung der durchschnittlichen Verluste über die gleichen Episodenzeitpunkte wie bei den Gesamtbelohnungen: 20, 40, 60, 80, 100 und 120 (mit möglichen Abweichungen bei einzelnen Agenten).

#### Agent 0

- Der durchschnittliche Verlust von Agent 0 zeigt einen **allmählichen Rückgang**, was darauf hindeutet, dass das Modell des Agenten besser optimiert wird und genauere Vorhersagen trifft.

#### Agent 1

- Ein ähnlicher Trend wie bei Agent 0 ist bei Agent 1 erkennbar. Der durchschnittliche Verlust **nimmt mit der Zeit ab**, was eine Verbesserung der Lernfähigkeit des Agenten bestätigt.

#### Agent 2

- Der durchschnittliche Verlust von Agent 2 **nimmt ebenfalls ab**, was auf eine effiziente Fehlerkorrektur und ein effektiveres Lernen hindeutet.

#### Agent 3

- Auch Agent 3 zeigt einen **Rückgang der durchschnittlichen Verluste**, was darauf hindeutet, dass die Anpassung des Agenten im Laufe der Trainingsphasen effektiv ist.

## Lernrate pro Episode

### Episodenfortschritt

Die Lernrate wurde zu den folgenden Episodenzeitpunkten untersucht: 0, 50, 100, 150, 200, 250, 300 und 350.

- **Alle Agenten** behalten eine **konstante Lernrate** über die beobachteten Episoden hinweg. Diese Stabilität sorgt für eine konsistente Anpassungsgeschwindigkeit des Modells und ist wichtig für die Synchronisation zwischen den verschiedenen Agenten während des Trainings.

## Trendanalyse: Gesamtbelohnungen und Verluste

Diese detailliertere Trendanalyse betrachtet die Gesamtbelohnungen und durchschnittlichen Verluste jedes Agenten, wobei insbesondere rollierende Mittelwerte und Mediane betrachtet werden.

### Agent 0

- **Gesamtbelohnungen:** Deutlich **steigender Trend**. Der rollierende Mittelwert (Rolling Mean) und der rollierende Median (Rolling Median) bestätigen die positive Entwicklung der Belohnungen.
- **Durchschnittlicher Verlust:** **Abnehmend**. Der Verlust reduziert sich signifikant, was eine Verbesserung der Performance des Agenten belegt.

### Agent 1

- **Gesamtbelohnungen:** **Stetiger Anstieg** über die Episoden hinweg. Sowohl der Rolling Mean als auch der Median bestätigen diese aufwärtsgerichtete Tendenz.
- **Durchschnittlicher Verlust:** **Abnehmende Tendenz** bestätigt eine effektive Lernkurve des Agenten.

### Agent 2

- **Gesamtbelohnungen:** Eine klare **Tendenz zum Anstieg** der Belohnungen. Unterstützt wird dies durch die Analyse des Rolling Mean und Median.
- **Durchschnittlicher Verlust:** Der Verlust **verbessert sich kontinuierlich**, was auf ein stabiles Lernverhalten hinweist.

### Agent 3

- **Gesamtbelohnungen:** Der **Aufwärtstrend** ist deutlich erkennbar und wird durch die Analyse der Rolling-Daten bestätigt.
- **Durchschnittlicher Verlust:** Der **stetige Rückgang** der Verluste unterstreicht die anhaltende Anpassung des Agenten.

## Fazit

Das Training aller vier Agenten zeigt **positive Entwicklungen** sowohl in Bezug auf die Gesamtbelohnungen als auch auf die durchschnittlichen Verluste. Alle Agenten zeigen **konsistente Verbesserungen** über die Episoden hinweg, was auf eine effektive Trainingsstrategie hindeutet. Die **stabile Lernrate** unterstützt diese positiven Trends und sichert eine Synchronisation der Modellkomponenten.

Für weiterführende Analysen wird empfohlen, die Daten **regelmäßig zu überprüfen**, um die optimale Leistung über längere Trainingsperioden zu gewährleisten und frühzeitig auf mögliche Probleme reagieren zu können. Eine visuelle Darstellung der Daten (z.B. Graphen) kann das Verständnis des Fortschritts zusätzlich verbessern.

## Nächste Schritte

- **Visualisierung:** Generieren von Grafiken, um die Trends besser zu veranschaulichen.
- **Hypereparameter-Optimierung:** Feintuning der Trainingsparameter, um die Leistung weiter zu optimieren.
- **Validierung:** Durchführung von Validierungstests, um die Generalisierungsfähigkeit des Modells zu prüfen.

## Die Analysediagramme ind in der PDF zu finden.

