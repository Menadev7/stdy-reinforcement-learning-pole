import gymnasium as gym
import numpy as np
import time

EPISODIOS = 20000 
ALPHA = 0.1      
GAMMA = 0.99     
EPSILON = 1.0
EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.01
SHOW_PROGRESS_EVERY = 2000

BINS = [20, 20, 20, 20] 
min_values = [-4.8, -4.0, -0.418, -4.0]
max_values = [4.8, 4.0, 0.418, 4.0]

def discretize_state(state):
    discrete_state = []
    for i in range(len(state)):
        scaling = (state[i] + abs(min_values[i])) / (max_values[i] - min_values[i])
        new_val = int(round((BINS[i] - 1) * scaling))
        new_val = max(0, min(new_val, BINS[i] - 1))
        discrete_state.append(new_val)
    return tuple(discrete_state)

# --- Inicialización ---
q_table = np.zeros(BINS + [gym.make("CartPole-v1").action_space.n])

# ==============================================================================
# --- FASE 1: DEMOSTRACIÓN INICIAL (AGENTE SIN ENTRENAR) ---
# ==============================================================================
print("El agente se movera al azar y fallara inmediatamente.")
time.sleep(2)

env_render = gym.make("CartPole-v1", render_mode="human")
for i in range(3):
    print(f"Intento de agente sin entrenar #{i+1}")
    state_continuo, info = env_render.reset()
    done = False
    while not done:
        action = env_render.action_space.sample() 
        new_state_continuo, reward, terminated, truncated, info = env_render.step(action)
        done = terminated or truncated
        time.sleep(0.05)
        
        if terminated:
            for _ in range(25):
                env_render.step(0) 
                time.sleep(0.02)
        
env_render.close()
print("--- Fin de la demostracion inicial ---\n")


# ==============================================================================
# --- FASE 2: ENTRENAMIENTO RÁPIDO Y SILENCIOSO ---
# ==============================================================================
print("--- FASE 2: Iniciando entrenamiento acelerado... ---")
env_train = gym.make("CartPole-v1")

for episodio in range(EPISODIOS + 1):
    if episodio > 0 and episodio % SHOW_PROGRESS_EVERY == 0:
        print(f"Progreso: Episodio {episodio}/{EPISODIOS} | Epsilon: {EPSILON:.4f}")

    state_continuo, info = env_train.reset()
    state = discretize_state(state_continuo)
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < EPSILON:
            action = env_train.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        new_state_continuo, reward, terminated, truncated, info = env_train.step(action)
        done = terminated or truncated
        new_state = discretize_state(new_state_continuo)
        old_value = q_table[state + (action,)]
        next_max = np.max(q_table[new_state])
        new_q_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        q_table[state + (action,)] = new_q_value
        state = new_state

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

print("¡Entrenamiento finalizado!")
env_train.close()


# ==============================================================================
# --- FASE 3: DEMOSTRACIÓN FINAL (AGENTE EXPERTO) ---
# ==============================================================================
print("\n--- FASE 3: Demostración del agente ENTRENADO ---")
env_render = gym.make("CartPole-v1", render_mode="human")
for i in range(3): 
    print(f"Intento de demostracion experta #{i+1}")
    state_continuo, info = env_render.reset()
    state = discretize_state(state_continuo)
    done = False

    while not done:
        action = np.argmax(q_table[state])
        new_state_continuo, reward, terminated, truncated, info = env_render.step(action)
        done = terminated or truncated
        state = discretize_state(new_state_continuo)
        time.sleep(0.05)
        
        if terminated:
            print("Caida detectada Mostrando animacion completa...")
            for _ in range(25):
                env_render.step(0) 
                time.sleep(0.02)
env_render.close()