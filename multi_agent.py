"""
Multi agent projects from Udacity's DRLND
"""

from unityagents import UnityEnvironment
from drlnd.common.agents.utils import ReplayBuffer, ActionType
from drlnd.common.agents.maddpg import MADDPGAgent, AgentSpec
from drlnd.common.utils import get_next_results_directory, path_from_project_home, OrnsteinUhlenbeckProcess
from collections import deque
import click
import torch
import numpy as np
import os
import yaml


def get_unity_env(path: str):
    env = UnityEnvironment(file_name=path)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]   
    num_agents = len(env_info.agents)

    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    print(f"Num agents: {num_agents}\nAction dim: f{action_size}\nState dim: {state_size}")

    return env, brain_name, num_agents, action_size, state_size



@click.group()
def cli():
    pass


@cli.command()
@click.option("--n-episodes", type=int, default = 1, 
    help="""
    Number of episodes to train for.
    """)
@click.option("--note", type=str, default=None, 
    help="""
    Note to record to .txt file when results are saved.
    """)
def train(n_episodes, note):
    """Train a pair of agents to play tennis using the MADDPG algorithm.
    """
    env, brain_name, num_agents, action_size, state_size = \
        get_unity_env('./unity_environments/Tennis/Tennis_Linux/Tennis.x86_64')

    buffer = ReplayBuffer(
        action_size, 
        int(5e6), 
        256, 
        1234, 
        action_dtype = ActionType.CONTINUOUS
        )

    agent_specs = []
    for i in range(num_agents):
        agent_specs.append(AgentSpec(state_size, action_size))
    agent = MADDPGAgent(agent_specs, buffer, hidden_layer_size=128)

    episode_scores = deque(maxlen=100)
    on_policy_scores = deque(maxlen=100)
    average_scores = []
    on_policy_averages = []
    online = lambda x: (x%10 == 0)
    learning_episodes = 0

    noise_fn_taper = 10000
    scale_next = 1.0

    for i in range(n_episodes):
        try:
            print(f"Episode: {i}")
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations
            scores = [0.0 for x in agent_specs]

            if not online(i):  
                # Include some noise in the action selection, which we linearly scale
                scale = max([1e-4, (1.0-(float(learning_episodes)/noise_fn_taper))])
                #noise_fn = lambda : torch.from_numpy(np.random.normal(loc=0.0, scale=scale, size=(action_size))).float()
                dt = np.random.choice([1e-2, 5e-2, 1e-1])
                theta = np.random.choice([0.1, 0.5, 1.0])
                noise_generator = OrnsteinUhlenbeckProcess([action_size], scale, dt=dt, theta=theta)
                noise_fn = lambda : torch.from_numpy(noise_generator.sample()).float()
                learning_episodes += 1
            else:
                scale = 0.0
                noise_fn = lambda : 0.0
                
            while True:
                actions = (
                    agent.act(
                        torch.from_numpy(state).float(),
                        policy_suppression = (1.0 - scale), 
                        noise_func = noise_fn)
                        )
                env_info = env.step([x.numpy() for x in actions])[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                if not online(i):
                    agent.replay_buffer.add(
                        np.concatenate(state), 
                        np.concatenate(actions), 
                        rewards, 
                        np.concatenate(next_states), 
                        dones)
                    agent.learn(0.99)

                scores = [scores[x] + rewards[x] for x in range(num_agents)]
                state = next_states
                if np.any(dones):
                    break

            if not online(i):
                episode_scores.append(max(scores))
                average_scores.append(np.mean(episode_scores))
            else:
                on_policy_scores.append(scores)
                on_policy_averages.append(np.mean(on_policy_scores))

            print('Total score this episode: {}'.format(env_info.rewards))
        except KeyboardInterrupt:
            print("Interrupted training, will save weights now")
            break

    results_directory = get_next_results_directory()
    agent.save_weights(results_directory)
    np.savetxt(os.path.join(results_directory, 'scores.txt'), average_scores)
    np.savetxt(os.path.join(results_directory, 'on_policy_scores.txt'), on_policy_averages)
    if note is not None:
        with open(os.path.join(results_directory, 'note.txt'), 'w') as f:
            f.write(note)
    params = {
        'noise_fn': repr(noise_fn),
        'noise_fn_taper': noise_fn_taper, 
        'taper': 'linear'}
    with open(os.path.join(results_directory, 'params.yml'), 'w') as f:
        yaml.dump(params, f)


@cli.command()
@click.option("--weights-path", type=str, default=None, 
    help="""
    Path to the directory containing the trained weights of the agents networks.
    Can be none, in which case the pre-trained weights in resources are used.
    """)
@click.option('--n-episodes', type=int, default=1, 
    help = """
    Number of episodes to train an agent for.
    """)
def run(weights_path: str, n_episodes: int):
    """Initialise an agent using pre-trained network weights and observe the 
    agent's interaction with the environment.
    """
    env, brain_name, num_agents, action_size, state_size = \
        get_unity_env('./unity_environments/Tennis/Tennis_Linux/Tennis.x86_64')

    agent_specs = []
    for i in range(num_agents):
        agent_specs.append(AgentSpec(state_size, action_size))
    agent = MADDPGAgent(agent_specs, None, hidden_layer_size=128)

    if weights_path is None:
        weights_path = path_from_project_home('./resources/solved_weights/')
    agent.load_weights(weights_path)

    for i in range(n_episodes):
        print(f"Episode: {i}")
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations

        score = 0.0
        for i in range(500):
            actions = (
                agent.act(
                    torch.from_numpy(states).float(), 
                    policy_suppression=1.0)
                    )
            env_info = env.step([x.numpy() for x in actions])[brain_name]
            states = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done

            score += env_info.rewards[0]
            if np.any(done):
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))


if __name__ == "__main__":
    os.environ['PROJECT_HOME'] = os.getcwd()
    cli()