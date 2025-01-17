import pygame
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import math
import logging

# -------------------- KONSTANTEN --------------------
# Konstanten für die Größe des Gitters und des Fensters
GRID_COLS = 20
GRID_ROWS = 20
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

# Berechnung der Gitterzellengröße
GRID_SIZE = min(WINDOW_WIDTH // GRID_COLS, WINDOW_HEIGHT // GRID_ROWS)

# Farben für die Agenten
AGENT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Farben für die Ressourcen
RESOURCE_COLORS = {
    0: (200, 200, 200),  # Leer
    1: (100, 100, 100),  # Erz
    2: (0, 150, 0),     # Baum
    3: (0, 0, 255),     # Wasser
    4: (0, 200, 100),   # Pflanze
    5: (255, 0, 255),   # Hindernis
    6: (255, 255, 0),   # Neue Ressource
    7: (255, 0, 0)      # Ziel
}

# -------------------- KONSTANTEN --------------------
# Lokale Sichtweite des Agenten
LOCAL_VIEW_SIZE = 1

# Anzahl der Aktionen: 0: N, 1: S, 2: W, 3: O, 4: Sammeln
NUM_ACTIONS = 5

# Typen der Ressourcen
RESOURCE_TYPES = [1, 2, 3, 4, 6]

# Dimension des Zustandsraums
STATE_DIM = (2 * LOCAL_VIEW_SIZE + 1) ** 2 + len(RESOURCE_TYPES) + len(RESOURCE_TYPES) + 1 + 1

# Häufigkeit der Visualisierung
VISUALIZATION_FREQUENCY = 15

# Rate des Wiedererscheinens der Ressourcen
RESOURCE_RESPAWN_RATE = 0.02

# Anfangsenergie der Agenten
INITIAL_ENERGY = 200

# Rate des Energieverlusts
ENERGY_DEPLETION_RATE = 1

# Menge der Energieaufladung durch Ressourcen
RESOURCE_RECHARGE_AMOUNT = 10

# Belohnungen für verschiedene Aktionen
REWARD_RESOURCE = 1
REWARD_STEP = -0.05
REWARD_DEAD = -2
REWARD_RARE_RESOURCE = 2
REWARD_COLLISION = -1
REWARD_IDLE = -0.1  # Strafe für das "Nichtstun"
REWARD_GOAL = 10    # Belohnung für das Erreichen des Ziels

# Dichte der Hindernisse
OBSTACLE_DENSITY = 0.05

# Anzahl der Agenten
NUM_AGENTS = 4

# Typen der Agenten
AGENT_TYPES = {
    0: {"speed": 1, "carry_capacity": 1},  # Standard
    1: {"speed": 2, "carry_capacity": 1},  # Schnell
    2: {"speed": 1, "carry_capacity": 2},  # Träger
    3: {"speed": 1, "carry_capacity": 1}   # Standard
}

# Reichweite der Kommunikation
COMMUNICATION_RANGE = 5

# Wahrscheinlichkeit der Kommunikation
COMMUNICATION_PROBABILITY = 0.2

# ----------------------------------------------------
# Hilfsfunktion für das Zeichnen des Agenten
def draw_agent(screen, agent_pos, agent_index):
    """
    Zeichnet einen Agenten auf dem Bildschirm.

    Args:
        screen (pygame.Surface): Der Bildschirm, auf dem gezeichnet wird.
        agent_pos (tuple): Die Position des Agenten.
        agent_index (int): Der Index des Agenten.
    """
    agent_x = agent_pos[1] * GRID_SIZE
    agent_y = agent_pos[0] * GRID_SIZE
    pygame.draw.rect(screen, AGENT_COLORS[agent_index % len(AGENT_COLORS)], (agent_x, agent_y, GRID_SIZE, GRID_SIZE))

def draw_grid(screen, env):
    """
    Zeichnet das Gitter und die Agenten auf dem Bildschirm.

    Args:
        screen (pygame.Surface): Der Bildschirm, auf dem gezeichnet wird.
        env (Environment): Die Umgebung, die gezeichnet wird.
    """
    screen.fill((255, 255, 255))
    for i in range(env.size):
        for j in range(env.size):
            x = j * GRID_SIZE
            y = i * GRID_SIZE
            resource_type = env.grid[i, j]
            pygame.draw.rect(screen, RESOURCE_COLORS[resource_type], (x, y, GRID_SIZE, GRID_SIZE))
    for i, agent_pos in enumerate(env.agent_positions):
        draw_agent(screen, agent_pos, i)
    pygame.display.flip()

# Funktion für die Berechnung des Zustandsraums
def get_state_from_grid(env, agent_index, local_view_size=1):
    """
    Berechnet den Zustandsraum für einen Agenten basierend auf der lokalen Sichtweite.

    Args:
        env (Environment): Die Umgebung.
        agent_index (int): Der Index des Agenten.
        local_view_size (int): Die lokale Sichtweite des Agenten.

    Returns:
        np.array: Der Zustandsraum des Agenten.
    """
    x, y = env.agent_positions[agent_index]
    start_x = max(0, x - local_view_size)
    end_x = min(env.size, x + local_view_size + 1)
    start_y = max(0, y - local_view_size)
    end_y = min(env.size, y + local_view_size + 1)
    local_grid = env.grid[start_x:end_x, start_y:end_y].flatten()
    pad_len = (2 * local_view_size + 1) ** 2 - len(local_grid)
    local_grid = np.pad(local_grid, (0, pad_len), 'constant', constant_values=0)

    resource_distances = []
    for resource_type in RESOURCE_TYPES:
        resource_positions = np.argwhere(env.grid == resource_type)
        if resource_positions.size > 0:
            distances = np.sqrt(np.sum((resource_positions - np.array(env.agent_positions[agent_index]))**2, axis=1))
            min_distance = np.min(distances)
        else:
            min_distance = env.size * math.sqrt(2)
        resource_distances.append(min_distance)

    agent_locations = np.array(env.agent_positions)
    other_agents = np.concatenate([agent_locations[:agent_index], agent_locations[agent_index+1:]], axis=0)
    if other_agents.size > 0:
        distances_to_others = np.sqrt(np.sum((other_agents - np.array(env.agent_positions[agent_index]))**2, axis=1))
        min_distance_to_other = np.min(distances_to_others)
    else:
        min_distance_to_other = env.size * math.sqrt(2)

    state = np.concatenate((
        local_grid,
        np.array(list(env.inventories[agent_index].values())),
        np.array(resource_distances),
        np.array([env.energies[agent_index]]),
        np.array([min_distance_to_other])
    ))

    # Debugging: Dimension des Zustands
    # print(f"State Dimension: {len(state)}, Expected: {STATE_DIM}")
    return state

class DualReplayBuffer:
    """
    Eine Klasse, die einen dualen Wiederholungspuffer implementiert.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):
        """
        Initialisiert den dualen Wiederholungspuffer.

        Args:
            capacity (int): Die Kapazität des Puffers.
            alpha (float): Der Exponent für die Prioritätenberechnung.
            beta_start (float): Der Anfangswert für Beta.
            beta_end (float): Der Endwert für Beta.
            beta_frames (int): Die Anzahl der Frames, über die Beta ansteigt.
        """
        self.capacity = capacity
        self.positive_buffer = deque(maxlen=capacity)
        self.negative_buffer = deque(maxlen=capacity)
        self.positive_priorities = deque(maxlen=capacity)
        self.negative_priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon_priority = 1e-6

    def push(self, state, action, reward, next_state, done, error):
        """
        Fügt einen neuen Eintrag in den Puffer hinzu.

        Args:
            state: Der aktuelle Zustand.
            action: Die ausgeführte Aktion.
            reward: Die Belohnung für die Aktion.
            next_state: Der nächste Zustand.
            done: Ob die Episode beendet ist.
            error: Der Fehler der Vorhersage.
        """
        max_priority = max(self.positive_priorities, default=1) if self.positive_buffer else 1
        if reward > 0:
            self.positive_buffer.append((state, action, reward, next_state, done))
            self.positive_priorities.append(max_priority)
        else:
            self.negative_buffer.append((state, action, reward, next_state, done))
            self.negative_priorities.append(max_priority)

    def _calculate_priorities(self, batch_indices, priorities):
        """
        Berechnet die Prioritäten für eine Stichprobe von Indizes.

        Args:
            batch_indices: Die Indizes der Stichprobe.
            priorities: Die Prioritäten der Stichprobe.

        Returns:
            np.array: Die Prioritäten der Stichprobe.
        """
        return np.array([priorities[i] for i in batch_indices])

    def update_priorities(self, indices, errors, positive=True):
        """
        Aktualisiert die Prioritäten für eine Stichprobe von Indizen.

        Args:
            indices: Die Indizes der Stichprobe.
            errors: Die Fehler der Vorhersagen.
            positive (bool): Ob die Prioritäten für positive Erfahrungen aktualisiert werden sollen.
        """
        priorities = self.positive_priorities if positive else self.negative_priorities
        for idx, error in zip(indices, errors):
            priorities[idx] = (abs(error) + self.epsilon_priority)**self.alpha

    def _calculate_probabilities(self, priorities):
        """
        Berechnet die Wahrscheinlichkeiten für die Stichprobenauswahl.

        Args:
            priorities: Die Prioritäten der Stichprobe.

        Returns:
            np.array: Die Wahrscheinlichkeiten der Stichprobenauswahl.
        """
        priorities = np.array(priorities, dtype=np.float64)
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()
        return probabilities

    def sample(self, batch_size, positive=True):
        """
        Erstellt eine Stichprobe aus dem Puffer.

        Args:
            batch_size: Die Größe der Stichprobe.
            positive (bool): Ob die Stichprobe aus positiven Erfahrungen sein soll.

        Returns:
            tuple: Die Stichprobe und die zugehörigen Gewichte.
        """
        buffer = self.positive_buffer if positive else self.negative_buffer
        priorities = self.positive_priorities if positive else self.negative_priorities
        if len(buffer) < batch_size:
            return None, None, None, None, None, None, None
        probabilities = self._calculate_probabilities(priorities)
        indices = np.random.choice(len(buffer), batch_size, p=probabilities, replace=False)
        batch = [buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        self.frame += 1
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1, self.frame / self.beta_frames)
        priorities = self._calculate_priorities(indices, priorities)
        weights = (len(buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return states, actions, rewards, next_states, dones, indices, torch.FloatTensor(weights).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __len__(self):
        """
        Gibt die Länge des Puffers zurück.

        Returns:
            int: Die Länge des Puffers.
        """
        return len(self.positive_buffer) + len(self.negative_buffer)

class NoisyLinear(nn.Module):
    """
    Eine Klasse, die eine noisy Linearschicht implementiert.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        """
        Initialisiert die noisy Linearschicht.

        Args:
            in_features (int): Die Anzahl der Eingabemerkmale.
            out_features (int): Die Anzahl der Ausgabemerkmale.
            std_init (float): Die Standardabweichung der Initialisierung.
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Setzt die Parameter der Schicht zurück.
        """
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """
        Setzt das Rauschen der Schicht zurück.
        """
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        """
        Führt eine Vorwärtsdurchführung der noisy Linearschicht durch.

        Args:
            input (torch.Tensor): Der Eingabezustand.

        Returns:
            torch.Tensor: Der Ausgabezustand.
        """
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class QNetwork(nn.Module):
    """
    Eine Klasse, die ein neuronales Netzwerk für die Q-Wert-Schätzung implementiert.
    """
    def __init__(self, input_dim, num_actions):
        """
        Initialisiert das neuronale Netzwerk.

        Args:
            input_dim (int): Die Dimension des Eingaberaums.
            num_actions (int): Die Anzahl der Aktionen.
        """
        super(QNetwork, self).__init__()
        self.fc1 = NoisyLinear(input_dim, 256)
        self.fc2 = NoisyLinear(256, 256)
        self.fc3 = NoisyLinear(256, num_actions)

    def forward(self, x):
        """
        Führt eine Vorwärtsdurchführung des neuronalen Netzwerks durch.

        Args:
            x (torch.Tensor): Der Eingabezustand.

        Returns:
            torch.Tensor: Die Q-Werte für jede Aktion.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def reset_noise(self):
        """
        Setzt das Rauschen des neuronalen Netzwerks zurück.
        """
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

class DDQNAgent:
    """
    Eine Klasse, die einen DDQN-Agenten implementiert.
    """
    def __init__(self, input_dim, num_actions, learning_rate=0.00025, gamma=0.99, batch_size=64, buffer_capacity=100000, update_target_freq=1000, tau=0.001, learning_rate_decay=0.000002):
        """
        Initialisiert den DDQN-Agenten.

        Args:
            input_dim (int): Die Dimension des Eingaberaums.
            num_actions (int): Die Anzahl der Aktionen.
            learning_rate (float): Die Lernrate.
            gamma (float): Der Diskontierungsfaktor.
            batch_size (int): Die Größe der Stichprobe.
            buffer_capacity (int): Die Kapazität des Puffers.
            update_target_freq (int): Die Häufigkeit der Aktualisierung des Zielnetzwerks.
            tau (float): Der Wert für die weiche Aktualisierung.
            learning_rate_decay (float): Der Verfallswert für die Lernrate.
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(input_dim, num_actions).to(self.device)
        self.target_network = QNetwork(input_dim, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = DualReplayBuffer(buffer_capacity)
        self.update_target_freq = update_target_freq
        self.steps = 0
        self.log_file = open("tiefsee_train_log.txt", "w")
        self.tau = tau
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.moving_avg_loss = 0
        self.best_reward = -float('inf')
        self.best_model_path = "tiefsee_best_model.pth"
        self.training = True
        self.current_episode = 0

    def choose_action(self, state):
        """
        Wählt eine Aktion basierend auf dem aktuellen Zustand.

        Args:
            state (np.array): Der aktuelle Zustand.

        Returns:
            int: Die gewählte Aktion.
        """
        if self.training:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()

    def learn(self):
        """
        Führt einen Lernschritt durch.

        Returns:
            float: Der Verlust des Lernschritts.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.steps += 1
        positive = self.steps % 2 == 0
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, positive=positive)

        if states is None:
            return None

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        weights = weights.unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            best_actions = torch.argmax(self.q_network(next_states), dim=1).unsqueeze(1)
            next_q_values_target = next_q_values.gather(1, best_actions)

        target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values_target * (1 - dones)

        errors = (target_q_values - q_values).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, errors, positive=positive)

        loss = (F.mse_loss(q_values, target_q_values, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.q_network.reset_noise()
        self.target_network.reset_noise()

        if self.steps % self.update_target_freq == 0:
            self._soft_update_target_network()

        return loss.item()

    def _soft_update_target_network(self):
        """
        Führt eine weiche Aktualisierung des Zielnetzwerks durch.
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def log(self, episode, reward, loss):
        """
        Protokolliert die Trainingsinformationen.

        Args:
            episode (int): Die aktuelle Episode.
            reward (float): Die Belohnung der Episode.
            loss (float): Der Verlust der Episode.
        """
        self.log_file.write(f"Episode: {episode}, Reward: {reward}, Loss: {loss}, LR: {self.learning_rate}\n")

    def save(self, filepath):
        """
        Speichert das Modell.

        Args:
            filepath (str): Der Pfad zur Speicherdatei.
        """
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """
        Lädt das Modell.

        Args:
            filepath (str): Der Pfad zur Ladedatei.
        """
        checkpoint = torch.load(filepath, weights_only=True)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_network.load_state_dict(self.q_network.state_dict())

    def close_log(self):
        """
        Schließt die Protokolldatei.
        """
        self.log_file.close()

    def adjust_learning_rate(self, loss):
        """
        Passt die Lernrate an.

        Args:
            loss (float): Der Verlust der Episode.
        """
        if self.moving_avg_loss == 0:
            self.moving_avg_loss = loss
        else:
            self.moving_avg_loss = 0.95 * self.moving_avg_loss + 0.05 * loss
        self.learning_rate = max(0.00001, self.learning_rate - self.learning_rate_decay)
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate

    def save_best_model(self, reward):
        """
        Speichert das beste Modell basierend auf der Belohnung.

        Args:
            reward (float): Die Belohnung der Episode.
        """
        if reward > self.best_reward:
            self.best_reward = reward
            self.save(self.best_model_path)
            print(f"Best model updated, new reward {reward}")

    def train_mode(self):
        """
        Schaltet den Agenten in den Trainingsmodus.
        """
        self.training = True

    def eval_mode(self):
        """
        Schaltet den Agenten in den Evaluierungsmodus.
        """
        self.training = False

class Environment:
    """
    Eine Klasse, die die Umgebung für die Agenten implementiert.
    """
    def __init__(self, size, resources, num_agents=1):
        """
        Initialisiert die Umgebung.

        Args:
            size (int): Die Größe des Gitters.
            resources (dict): Die Ressourcen und ihre Dichten.
            num_agents (int): Die Anzahl der Agenten.
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.agent_positions = [(random.randint(0, size - 1), random.randint(0, size - 1)) for _ in range(num_agents)]
        self.inventories = [{i: 0 for i in resources} for _ in range(num_agents)]
        self.resources = resources
        self.energies = [INITIAL_ENERGY for _ in range(num_agents)]
        self.num_agents = num_agents
        self.agent_types = [random.choice(list(AGENT_TYPES.keys())) for _ in range(num_agents)]
        self._populate_resources()
        self._populate_obstacles()
        self.communications = [{} for _ in range(num_agents)]
        self.goals = [(random.randint(0, size - 1), random.randint(0, size - 1)) for _ in range(num_agents)]

    def _populate_resources(self):
        """
        Befüllt das Gitter mit Ressourcen.
        """
        for resource_type, density in self.resources.items():
            num_resources = int(self.size * self.size * density)
            for _ in range(num_resources):
                while True:
                    x = random.randint(0, self.size - 1)
                    y = random.randint(0, self.size - 1)
                    if self.grid[x, y] == 0:
                        self.grid[x, y] = resource_type
                        break

    def _populate_obstacles(self):
        """
        Befüllt das Gitter mit Hindernissen.
        """
        num_obstacles = int(self.size * self.size * OBSTACLE_DENSITY)
        for _ in range(num_obstacles):
            while True:
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = 5  # 5 ist die ID für Hindernisse
                    break

    def _move_obstacles(self):
        """
        Bewegt die Hindernisse dynamisch.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 5:
                    direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                    new_x, new_y = (i + direction[0]) % self.size, (j + direction[1]) % self.size
                    if self.grid[new_x, new_y] == 0:
                        self.grid[new_x, new_y] = 5
                        self.grid[i, j] = 0

    def _move_resources(self):
        """
        Bewegt die Ressourcen dynamisch.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] in RESOURCE_TYPES:
                    direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                    new_x, new_y = (i + direction[0]) % self.size, (j + direction[1]) % self.size
                    if self.grid[new_x, new_y] == 0:
                        self.grid[new_x, new_y] = self.grid[i, j]
                        self.grid[i, j] = 0

    def step(self, actions):
        """
        Führt einen Schritt in der Umgebung aus.

        Args:
            actions (list): Die Aktionen der Agenten.

        Returns:
            tuple: Die nächsten Zustände, Belohnungen und ob die Episode beendet ist.
        """
        all_next_states = []
        all_rewards = []
        all_dones = []
        for agent_index, action in enumerate(actions):
            x, y = self.agent_positions[agent_index]
            agent_type = self.agent_types[agent_index]
            speed = AGENT_TYPES[agent_type]["speed"]

            if action == 0:  # Norden
                new_x, new_y = max(0, x - speed), y
            elif action == 1:  # Süden
                new_x, new_y = min(self.size - 1, x + speed), y
            elif action == 2:  # Westen
                new_x, new_y = x, max(0, y - speed)
            elif action == 3:  # Osten
                new_x, new_y = x, min(self.size - 1, y + speed)
            elif action == 4:  # Sammeln
                new_x, new_y = x, y
                resource_type = self.grid[x, y]
                if resource_type != 0 and resource_type != 5:  # Verhindern von Einsammeln von Hindernissen
                    carry_capacity = AGENT_TYPES[agent_type]["carry_capacity"]
                    self.inventories[agent_index][resource_type] = min(self.inventories[agent_index][resource_type] + 1, carry_capacity)
                    if resource_type == 4:
                        self.energies[agent_index] = min(INITIAL_ENERGY, self.energies[agent_index] + RESOURCE_RECHARGE_AMOUNT * 2)
                    else:
                        self.energies[agent_index] = min(INITIAL_ENERGY, self.energies[agent_index] + RESOURCE_RECHARGE_AMOUNT)
                    self.grid[x, y] = 0

            collision_reward = 0
            if self.grid[new_x, new_y] != 5:
                for other_agent_index, other_pos in enumerate(self.agent_positions):
                    if other_agent_index != agent_index and (new_x, new_y) == other_pos:
                        collision_reward = REWARD_COLLISION
                        break
                self.agent_positions[agent_index] = (new_x, new_y)
            else:
                collision_reward = REWARD_COLLISION
            self._respawn_resources()
            self.energies[agent_index] = max(0, self.energies[agent_index] - ENERGY_DEPLETION_RATE)
            reward, done = self._get_reward(agent_index, collision_reward)
            all_next_states.append(self.get_state(agent_index))
            all_rewards.append(reward)
            all_dones.append(done)
            self._communicate(agent_index)

        self._move_obstacles()
        self._move_resources()

        return all_next_states, all_rewards, all_dones



    def _communicate(self, agent_index):
        """
        Lässt den Agenten mit anderen Agenten kommunizieren.

        Args:
            agent_index (int): Der Index des Agenten.
        """
        x, y = self.agent_positions[agent_index]
        for other_agent_index in range(self.num_agents):
            if other_agent_index != agent_index:
                ox, oy = self.agent_positions[other_agent_index]
                distance = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                if distance <= COMMUNICATION_RANGE and random.random() < COMMUNICATION_PROBABILITY:
                    self.communications[agent_index][other_agent_index] = {"position": (ox, oy), "resources": self.inventories[other_agent_index]}

    def _respawn_resources(self):
        """
        Lässt Ressourcen wieder erscheinen.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0 and random.random() < RESOURCE_RESPAWN_RATE:
                    resource_types = [r for r in self.resources if r != 0]
                    if resource_types:
                        self.grid[i, j] = random.choice(resource_types)

    def _get_reward(self, agent_index, collision_reward):
        """
        Berechnet die Belohnung für einen Agenten.

        Args:
            agent_index (int): Der Index des Agenten.
            collision_reward (float): Die Belohnung für eine Kollision.

        Returns:
            tuple: Die Belohnung und ob die Episode beendet ist.
        """
        if self.energies[agent_index] <= 0:
            return REWARD_DEAD, True

        reward = 0
        for resource_type in self.inventories[agent_index]:
            if self.inventories[agent_index][resource_type] > 0:
                if resource_type == 4:
                    reward += REWARD_RARE_RESOURCE
                else:
                    reward += REWARD_RESOURCE
        reward += REWARD_STEP
        reward += collision_reward
        reward += REWARD_IDLE  # Strafe für das "Nichtstun"

        # Belohnung für das Erreichen des Ziels
        if self.agent_positions[agent_index] == self.goals[agent_index]:
            reward += REWARD_GOAL
            self.goals[agent_index] = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))  # Neues Ziel setzen

        return reward, False

    def get_state(self, agent_index):
        """
        Gibt den Zustand eines Agenten zurück.

        Args:
            agent_index (int): Der Index des Agenten.

        Returns:
            np.array: Der Zustand des Agenten.
        """
        return get_state_from_grid(self, agent_index)

# Pygame Initialisierung
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("RL Training")
clock = pygame.time.Clock()

# Trainingsloop
# Trainingsloop
env = Environment(size=GRID_COLS, resources={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.05, 6: 0.05}, num_agents=NUM_AGENTS)
state_dim = STATE_DIM
action_dim = NUM_ACTIONS
agents = [DDQNAgent(state_dim, action_dim) for _ in range(NUM_AGENTS)]

# Laden des zuletzt gespeicherten Modells zu Beginn des Trainings
for i, agent in enumerate(agents):
    model_path = f"model_agent_{i}.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model for agent {i} from {model_path}")

num_episodes = 5001
running = True
for episode in range(num_episodes):
    print(f"---Episode {episode}---")
    all_states = [env.get_state(i) for i in range(NUM_AGENTS)]
    episode_rewards = [0 for _ in range(NUM_AGENTS)]
    episode_losses = [[] for _ in range(NUM_AGENTS)]
    dones = [False for _ in range(NUM_AGENTS)]
    for agent in agents:
        agent.current_episode = episode
    for step in range(200):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break
        if all(dones):
            break
        actions = [agents[i].choose_action(all_states[i]) for i in range(NUM_AGENTS)]
        next_states, rewards, dones = env.step(actions)
        for i in range(NUM_AGENTS):
            error = 0
            agents[i].replay_buffer.push(all_states[i], actions[i], rewards[i], next_states[i], dones[i], error)
            loss = agents[i].learn()
            if loss is not None:
                episode_losses[i].append(loss)
            episode_rewards[i] += rewards[i]
            all_states[i] = next_states[i]

        if step % VISUALIZATION_FREQUENCY == 0:
            draw_grid(screen, env)
        clock.tick(60)

    if not running:
        break
    for i in range(NUM_AGENTS):
        avg_loss = np.mean(episode_losses[i]) if episode_losses[i] else None
        agents[i].adjust_learning_rate(avg_loss if avg_loss is not None else 0)
        agents[i].log(episode, episode_rewards[i], avg_loss)
        print(f"Agent {i}: Total reward: {episode_rewards[i]:.2f}, Average Loss: {avg_loss:.4f}, LR: {agents[i].learning_rate:.6f}")
        agents[i].save_best_model(episode_rewards[i])

    # Regelmäßiges Speichern des aktuellen Modells
    if episode % 100 == 0 and episode > 0:
        for i, agent in enumerate(agents):
            agent.save(f"model_agent_{i}.pth")
        print(f"Model at episode {episode} saved successfully")

    if episode % 1000 == 0 and episode > 0:
        print(f"Starting evaluation at episode {episode} ...")
        for agent in agents:
            agent.eval_mode()
        test_env = Environment(size=GRID_COLS, resources={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.05, 6: 0.05}, num_agents=NUM_AGENTS)
        test_rewards = [0 for _ in range(NUM_AGENTS)]
        test_dones = [False for _ in range(NUM_AGENTS)]
        test_states = [test_env.get_state(i) for i in range(NUM_AGENTS)]

        for test_step in range(200):
            if all(test_dones):
                break
            test_actions = [agents[i].choose_action(test_states[i]) for i in range(NUM_AGENTS)]
            test_states, test_rews, test_dones = test_env.step(test_actions)
            for i in range(NUM_AGENTS):
                test_rewards[i] += test_rews[i]

            if test_step % VISUALIZATION_FREQUENCY == 0:
                draw_grid(screen, test_env)
            clock.tick(60)
        for i in range(NUM_AGENTS):
            print(f"Evaluation at episode {episode} completed. Total test reward for agent {i}: {test_rewards[i]}")
        for agent in agents:
            agent.train_mode()

print("Training completed.")
print("Loading the best models ...")
for i, agent in enumerate(agents):
    agent.load(agent.best_model_path)
    agent.close_log()
pygame.quit()
