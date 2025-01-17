import pygame
import random
import numpy as np
from logic import DDQNAgent, Environment  # Importieren Sie die Klassen aus train.py

# Konstanten
GRID_SIZE = 40  
CELL_SIZE = 16  # Größe jeder Zelle in Pixeln
WINDOW_SIZE = GRID_SIZE * CELL_SIZE  # Gesamtgröße des Fensters
NUM_AGENTS = 10  # Anzahl der Agenten
STATE_DIM = 21  # Beispielwert, sollte basierend auf der tatsächlichen Implementierung angepasst werden
NUM_ACTIONS = 5  # Beispielwert, sollte basierend auf der tatsächlichen Implementierung angepasst werden
VISUALIZATION_FREQUENCY = 10
RESOURCE_COLORS = {7: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 6: (0, 255, 255)}  # Verschiedene Ressourcenfarben
COMMUNICATION_RANGE = 2  # Kommunikationsreichweite der Agenten

# Initialisieren von Pygame
pygame.init()

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("RL Training")
clock = pygame.time.Clock()  # FPS-Limiter initialisieren

# Laden der trainierten Modelle
# Initialisieren der Agenten mit eindeutigen IDs
agents = [DDQNAgent(STATE_DIM, NUM_ACTIONS, agent_id=i) for i in range(NUM_AGENTS)]

for i, agent in enumerate(agents):
    try:
        agent.load(f"model_agent_{i}_episode_4900.pth", weights_only=True)
    except FileNotFoundError:
        print(f"Model file for agent {i} not found. Skipping...")
        continue
    agent.eval_mode()

# Initialisieren der Umgebung und Agenten
env = Environment(size=GRID_SIZE, resources={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.05, 6: 0.05}, num_agents=NUM_AGENTS)

# Zielsetzung
target_pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
env.grid[target_pos[0], target_pos[1]] = 7  # 7 ist die ID für das Ziel

# Visualisierung
def draw_grid(screen, env, target_pos, cell_size=CELL_SIZE):
    """
    Zeichnet das Gitter der Umgebung, das Ziel und die beweglichen Hindernisse (BH).
    
    Args:
        screen (pygame.Surface): Der Bildschirm, auf dem gezeichnet wird.
        env (Environment): Die Umgebung.
        target_pos (tuple): Die Position des Ziels.
        cell_size (int): Die Größe einer Zelle im Gitter.
    """
    font = pygame.font.Font(None, 36)

    for x in range(env.size):
        for y in range(env.size):
            color = (230, 230, 230)  # Standardfarbe für leere Zellen
            if env.grid[x, y] != 0:
                color = RESOURCE_COLORS.get(env.grid[x, y], (200, 200, 200))  # Farben für Ressourcen

            pygame.draw.rect(screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))

            # Gitterlinien
            pygame.draw.rect(screen, (100, 100, 100), (y * cell_size, x * cell_size, cell_size, cell_size), 1)

            # „BH“-Hindernisse zeichnen
            if env.grid[x, y] == 5:  # Hindernis-ID 5 für bewegliche Hindernisse
                label = font.render("BH", True, (0, 0, 0))  # Schwarzer Text für "BH"
                screen.blit(label, (y * cell_size + cell_size // 4, x * cell_size + cell_size // 4))  # Position des Labels

    # Ziel markieren
    target_x, target_y = target_pos
    pygame.draw.rect(screen, (255, 0, 0), (target_y * cell_size, target_x * cell_size, cell_size, cell_size))
    label = font.render("Z", True, (255, 255, 255))
    screen.blit(label, (target_y * cell_size + cell_size // 4, target_x * cell_size + cell_size // 4))

def visualize_agents(screen, env, cell_size=CELL_SIZE):
    """
    Zeichnet die Agenten auf dem Bildschirm.

    Args:
        screen (pygame.Surface): Der Bildschirm, auf dem gezeichnet wird.
        env (Environment): Die Umgebung.
        cell_size (int): Die Größe einer Zelle im Gitter.
    """
    font = pygame.font.Font(None, 36)
    for i, agent_pos in enumerate(env.agent_positions):
        x, y = agent_pos[1] * cell_size, agent_pos[0] * cell_size
        pygame.draw.circle(screen, (255, 255, 255), (y + cell_size // 2, x + cell_size // 2), cell_size // 4)
        label = font.render(f"A{i+1}", True, (0, 0, 0))
        screen.blit(label, (y + cell_size // 4, x + cell_size // 4))


# Fokussierte Bewegung zur Zielperson
def calculate_next_move(agent_pos, target_pos, env):
    """
    Berechnet die nächste Bewegung eines Agenten in Richtung des Ziels und vermeidet Hindernisse und blockierte Zellen.

    Args:
        agent_pos (tuple): Die aktuelle Position des Agenten.
        target_pos (tuple): Die Zielposition.
        env (Environment): Die aktuelle Umgebung, um zu überprüfen, ob Hindernisse blockieren.

    Returns:
        tuple: Die nächste Position, die der Agent anstreben sollte.
    """
    x_diff = target_pos[0] - agent_pos[0]
    y_diff = target_pos[1] - agent_pos[1]
    possible_moves = []

    # Alle möglichen Bewegungen überprüfen
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_x, new_y = agent_pos[0] + dx, agent_pos[1] + dy
        if 0 <= new_x < env.size and 0 <= new_y < env.size:  # Prüfen, ob die neue Position innerhalb des Gitters liegt
            if env.grid[new_x, new_y] not in [5, 1, 2, 3, 4, 6]:  # Keine Hindernisse oder blockierte Ressourcen
                possible_moves.append((new_x, new_y))

    # Wählen Sie die beste Bewegung in Richtung des Ziels
    if not possible_moves:
        return agent_pos  # Wenn keine Bewegung möglich ist, bleibt der Agent an seiner Position

    # Wählen Sie die Bewegung, die dem Ziel am nächsten ist
    best_move = min(possible_moves, key=lambda pos: abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1]))
    return best_move


# Visualisierung der Agentenbewegungen
def highlight_agents_near_target(screen, env, target_pos, range=2, cell_size=CELL_SIZE):
    """
    Hebt Agenten hervor, die sich in der Nähe des Ziels befinden.

    Args:
        screen (pygame.Surface): Der Bildschirm, auf dem gezeichnet wird.
        env (Environment): Die Umgebung.
        target_pos (tuple): Die Position des Ziels.
        range (int): Die Reichweite, innerhalb der hervorgehoben wird.
        cell_size (int): Die Größe einer Zelle im Gitter.
    """
    for agent_pos in env.agent_positions:
        distance = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
        if distance <= range:
            x, y = agent_pos[1] * cell_size, agent_pos[0] * cell_size
            pygame.draw.rect(screen, (255, 165, 0), (y, x, cell_size, cell_size), 3)

# Agentenkooperation für die Zielsuche
explored_areas = set()

# Hauptschleife
running = True
steps = 0
found_target = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if found_target:
        break

    if steps % VISUALIZATION_FREQUENCY == 0:
        screen.fill((255, 255, 255))
        draw_grid(screen, env, target_pos)
        highlight_agents_near_target(screen, env, target_pos)
        visualize_agents(screen, env)
        pygame.display.flip()

    actions = []
    for i in range(NUM_AGENTS):
        if env.agent_positions[i] in explored_areas:
            actions.append(random.choice(range(NUM_ACTIONS)))  # Zufällige Bewegung, um das Grid zu erkunden
        else:
            next_move = calculate_next_move(env.agent_positions[i], target_pos, env)
            actions.append(next_move)

    next_states, rewards, dones = env.step(actions)

    # Debug-Ausgaben für Agentenpositionen
    print(f"Step {steps}: Agent positions - {env.agent_positions}")

    for agent_index in range(NUM_AGENTS):
        agent_pos = env.agent_positions[agent_index]
        explored_areas.add(agent_pos)

        # Überprüfung, ob Ziel gefunden
        if agent_pos == target_pos:
            found_target = True
            for other_agent_index in range(NUM_AGENTS):
                if agent_index != other_agent_index:
                    agents[agent_index].communicate(other_agent_index, True, target_pos)
            print(f"Target found by agent {agent_index} at {target_pos}!")
            break

    # Agenten-Kommunikation
    for agent_index in range(NUM_AGENTS):
        agent_pos = env.agent_positions[agent_index]
        for other_agent_index in range(NUM_AGENTS):
            if agent_index != other_agent_index:
                other_agent_pos = env.agent_positions[other_agent_index]
                distance = abs(agent_pos[0] - other_agent_pos[0]) + abs(agent_pos[1] - other_agent_pos[1])
                if distance <= COMMUNICATION_RANGE:
                    agents[agent_index].communicate(other_agent_index, False, explored_areas)

    # Agenten-Ressourcen
    for agent_index in range(NUM_AGENTS):
        agent_pos = env.agent_positions[agent_index]
        if env.grid[agent_pos[0], agent_pos[1]] in [1, 2, 3, 4, 6]:  # Ressourcen als Hindernisse
            continue  # Blockierte Bewegung

    steps += 1

    # Hier das FPS-Limit hinzufügen
    clock.tick(10)  # Setze die FPS auf 10 (verlangsamt das Spiel)

pygame.quit()
