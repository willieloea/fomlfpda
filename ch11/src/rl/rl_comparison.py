"""Known-model planning versus model-free learning.

World: states 0, 1, 2, 3, 4; start at 0, goal at 4.
Actions: left/right. Walls block movement. Entering goal pays +10;
every other move pays -1. An episode ends immediately at the goal.
"""

import random

ACTIONS = ("left", "right")
STATES = range(4)  # Nonterminal states; state 4 is terminal.
GAMMA = 0.9


def environment_step(state, action):
    """Execute a real action. Return (next_state, reward, done)."""
    if state == 4:
        raise ValueError("Episode has ended; reset to state 0.")
    movement = {"left": -1, "right": 1}[action]
    next_state = max(0, min(4, state + movement))
    done = next_state == 4
    return next_state, 10 if done else -1, done


# A GIVEN model: predictions for every possible state/action pair.
# It matches the real environment, but planning does not call environment_step.
# For a stochastic world, each pair would instead have a distribution of outcomes.
MODEL = {
    0: {"left": (0, -1, False), "right": (1, -1, False)},
    1: {"left": (0, -1, False), "right": (2, -1, False)},
    2: {"left": (1, -1, False), "right": (3, -1, False)},
    3: {"left": (2, -1, False), "right": (4, 10, True)},
}


def model_based(model):
    """Value iteration: plan using a known model, with zero real interactions."""
    values = {s: 0.0 for s in range(5)}

    def action_value(s, a):
        next_state, reward, done = model[s][a]  # KEY: predict an outcome.
        return reward + (0 if done else GAMMA * values[next_state])

    while True:
        updated = {s: max(action_value(s, a) for a in ACTIONS) for s in STATES}
        difference = max(abs(updated[s] - values[s]) for s in STATES)
        values.update(updated)
        if difference < 1e-10:
            break

    policy = {s: max(ACTIONS, key=lambda a: action_value(s, a)) for s in STATES}
    return policy, values


def model_free(step, episodes=2000):
    """Q-learning: learn action values from real, observed transitions."""
    rng = random.Random(7)
    q = {s: {a: 0.0 for a in ACTIONS} for s in range(5)}
    alpha, epsilon = 0.1, 0.3

    for _ in range(episodes):
        state = 0
        for _ in range(100):  # Safety cap; a capped episode is just reset.
            if rng.random() < epsilon:
                action = rng.choice(ACTIONS)
            else:
                best = max(q[state].values())
                action = rng.choice([a for a in ACTIONS if q[state][a] == best])

            next_state, reward, done = step(state, action)  # KEY: act and observe.
            target = reward + (0 if done else GAMMA * max(q[next_state].values()))
            q[state][action] += alpha * (target - q[state][action])
            state = next_state
            if done:
                break

    policy = {s: max(ACTIONS, key=lambda a: q[s][a]) for s in STATES}
    return policy, q


if __name__ == "__main__":
    planned_policy, values = model_based(MODEL)
    learned_policy, q = model_free(environment_step)
    print("State | Model-based action | Model-free action | V(s) | max Q(s,a)")
    for s in STATES:
        print(f"{s:5} | {planned_policy[s]:18} | {learned_policy[s]:17} | "
              f"{values[s]:4.2f} | {max(q[s].values()):.2f}")

