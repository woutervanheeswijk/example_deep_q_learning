# Needed for training the network
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

# Needed for animation
import matplotlib.pyplot as plt

def get_reward(bandit: float) -> tf.Tensor:
    """Generate reward for selected bandit"""
    reward = tf.random.normal([1], mean=bandit, stddev=1, dtype=tf.dtypes.float32)

    return reward


def plot(q_values: tf.Tensor, bandits: np.array) -> None:
    """Plot bar chart with selection probability per bandit"""
    q_values_plot = [
        q_values[0],
        q_values[1],
        q_values[2],
        q_values[3],
    ]
    bandit_plot = [
        bandits[0],
        bandits[1],
        bandits[2],
        bandits[3],
    ]
    width = 0.4
    x = np.arange(len(bandits))
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, q_values_plot, width, label="Q-values")
    ax.bar(x + width / 2, bandit_plot, width, label="True values")

    # Add labels and legend
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["1", "2", "3", "4"])

    plt.xlabel("Bandit")
    plt.ylabel("Value")
    plt.legend(loc="best")

    plt.show()

    return


def construct_q_network(state_dim: int, action_dim: int) -> keras.Model:
    """Construct the critic network with q-values per action as output"""
    inputs = layers.Input(shape=(state_dim,))  # input dimension
    hidden1 = layers.Dense(
        10, activation="relu", kernel_initializer=initializers.he_normal()
    )(inputs)
    hidden2 = layers.Dense(
        10, activation="relu", kernel_initializer=initializers.he_normal()
    )(hidden1)
    hidden3 = layers.Dense(
        10, activation="relu", kernel_initializer=initializers.he_normal()
    )(hidden2)
    q_values = layers.Dense(
        action_dim, kernel_initializer=initializers.Zeros(), activation="linear"
    )(hidden3)

    deep_q_network = keras.Model(inputs=inputs, outputs=[q_values])

    return deep_q_network


def mean_squared_error_loss(q_value: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
    """Compute mean squared error loss"""
    loss = 0.5 * (q_value - reward) ** 2

    return loss


if __name__ == "__main__":
    # Initialize parameters
    state = tf.constant([[1]])
    bandits = np.array([0.9, 1.2, 0.7, 1.0])
    state_dim = len(state)
    action_dim = len(bandits)
    exploration_rate = 0.1
    learning_rate = 0.01
    num_episodes = 10000

    # Construct Q-network
    q_network = construct_q_network(state_dim, action_dim)

    # Define optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for i in range(num_episodes + 1):
        with tf.GradientTape() as tape:
            # Obtain Q-values from network
            q_values = q_network(state)

            epsilon = np.random.rand()
            if epsilon <= exploration_rate:
                # Select random action
                action = np.random.choice(len(bandits))
            else:
                # Select action with highest q-value
                action = np.argmax(q_values)

            # Obtain reward from bandit
            reward = get_reward(bandits[action])

            # Obtain Q-value
            q_value = q_values[0, action]

            # Compute loss value
            loss_value = mean_squared_error_loss(q_value, reward)

            # Compute gradients
            grads = tape.gradient(loss_value[0], q_network.trainable_variables)

            # Apply gradients to update network weights
            opt.apply_gradients(zip(grads, q_network.trainable_variables))

            # Print console output
            if np.mod(i, 1000) == 0:
                print("\n======episode", i, "======")
                print("Q-values", ["%.3f" % n for n in q_values[0]])
                print(
                    "Rel. deviation",
                    [
                        "%.3f" % float((q_values[0, i] - bandits[i]) / bandits[i])
                        for i in range(len(q_values[0]))
                    ],
                )

    # Plot Q-values
    plot(q_values[0], bandits)
