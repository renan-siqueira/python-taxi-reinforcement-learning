import gym
import numpy as np
import os


def initialize_env(training=True):
    if training:
        return gym.make("Taxi-v3")
    return gym.make("Taxi-v3", render_mode='human')

def train_agent(env, Q, alpha, gamma, epsilon, episodes):
    total_epochs, total_rewards = 0, 0

    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        epochs, reward = 0, 0
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            old_value = Q[state, action]
            next_max = np.max(Q[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            Q[state, action] = new_value

            state = next_state
            epochs += 1

        total_rewards += reward
        total_epochs += epochs
    env.close()

    print(f"Resultados após {episodes} episódios:")
    print(f"Média de etapas por episódio: {total_epochs / episodes}")
    print(f"Média de recompensas por episódio: {total_rewards / episodes}")

def visualize_agent(env, Q, num_episodes=3):
    print("\nVisualizando o comportamento do agente após treinamento:")
    for _ in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        print("####### NOVO EPISÓDIO #########")

        while not done:
            action = np.argmax(Q[state])
            next_state, _, done, _, _ = env.step(action)
            env.render()
            state = next_state
            if isinstance(state, tuple):
                state = state[0]
    env.close()

def save_model(Q, file_name="q_table.npy"):
    np.save(file_name, Q)

def load_model(filename="q_table.npy"):
    if os.path.exists(filename):
        return np.load(filename)
    else:
        raise ValueError("Model file not found!")

def main():
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.2
    episodes = 10000

    choice = input("Do you want to train (T) or visualize (V)? ").lower()

    if choice == 't':
        env = initialize_env()
        Q = np.zeros([env.observation_space.n, env.action_space.n])
        train_agent(env, Q, alpha, gamma, epsilon, episodes)
        save = input("Do you want to save the model? (Y/N) ").lower()
        if save == 'y':
            save_model(Q)
    elif choice == 'v':
        env = initialize_env(training=False)
        Q = load_model()
        if Q is not None:
            visualize_agent(env, Q)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
