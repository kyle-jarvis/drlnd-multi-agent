"""
Multi agent projects from Udacity's DRLND
"""

from unityagents import UnityEnvironment
from drlnd.common.agents.utils import (
    ReplayBuffer,
    ActionType,
    get_unity_env,
    AgentInventory,
    UnityEnvWrapper,
)
from drlnd.common.agents.maddpg import MADDPGAgent2
from drlnd.common.utils import (
    get_next_results_directory,
    path_from_project_home,
    OrnsteinUhlenbeckProcess,
)
from collections import deque, namedtuple
import click
import torch
import torch.nn.functional as F
import numpy as np
import os
import yaml


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--n-episodes",
    type=int,
    default=1,
    help="""
    Number of episodes to train for.
    """,
)
@click.option(
    "--note",
    type=str,
    default=None,
    help="""
    Note to record to .txt file when results are saved.
    """,
)
@click.option(
    "--weights-path",
    type=str,
    default=None,
    help="""
    Path to the directory containing the trained weights of the agents networks.
    Can be none, in which case the pre-trained weights in resources are used.
    """,
)
@click.option(
    "--env-id",
    type=click.Choice(['Tennis', 'Soccer'], case_sensitive=False),
    default="Tennis",
    help="""
    """,
)
def train(n_episodes, note, weights_path, env_id):
    """Train a pair of agents to play tennis using the MADDPG algorithm."""

    env, brain_spec = get_unity_env(
        os.path.join(
            os.environ["PROJECT_HOME"],
            f"./unity_environments/{env_id}/{env_id}_Linux/{env_id}.x86_64",
        )
    )

    brain_spec_list = list(brain_spec.values())
    print(env.brain_names)

    env_wrapper = UnityEnvWrapper(env, brain_spec_list, ActionType.DISCRETE)

    # This defines an important ordering which sets which indices of the policy
    # and critic networks correspond to the observations and actions of each agent.
    # This list is passed to the agent inventory, which assigns slices to agent specs
    # and also is passed to the buffer.observe_environment method.

    buffer = ReplayBuffer(
        int(5e6),
        256,
        1234,
        action_dtype=ActionType.DISCRETE,
        brain_agents=brain_spec_list,
    )

    def policy_activation(x):
        if len(x.shape) == 1:
            return F.gumbel_softmax(x.unsqueeze(0)).squeeze()
        return F.gumbel_softmax(x)

    if env_id is "Tennis":
        policy_network_kwargs = {"output_activation": lambda x: x}
    else:
        policy_network_kwargs = {"output_activation": policy_activation}

    agent_inventory = AgentInventory(brain_spec_list)
    agent = MADDPGAgent2(agent_inventory, buffer, hidden_layer_size=256, policy_network_kwargs=policy_network_kwargs)

    if weights_path is not None:
        agent.load_weights(weights_path)

    episode_scores = deque(maxlen=100)
    on_policy_scores = deque(maxlen=100)
    average_scores = []
    on_policy_averages = []
    online = lambda x: (x % 10 == 0)
    learning_episodes = 0

    noise_fn_taper = 1000
    scale_next = 1.0
    exception_raised = None

    def make_noise_generator(size, scale, dt, theta):
        noise_generator = OrnsteinUhlenbeckProcess(
            [size], scale, dt, theta
            )
        noise_callable = lambda: torch.from_numpy(noise_generator.sample()).float()

        return noise_callable

    
    def no_noise_callable(size):
            return lambda: torch.zeros(size)

    no_noise = {brain.name: no_noise_callable(brain.action_size) for brain in brain_spec_list}            

    for i in range(n_episodes):
        try:
            print(f"Episode: {i}")
            env_wrapper.reset(train_mode=True)
            scores = [0.0 for x in range(agent_inventory.num_agents)]

            if not online(i):
                # Include some noise in the action selection, which we linearly scale
                scale = max([1e-4, (1.0 - (float(learning_episodes) / noise_fn_taper))])
                # noise_fn = lambda : torch.from_numpy(np.random.normal(loc=0.0, scale=scale, size=(action_size))).float()
                dt = np.random.choice([1e-2, 5e-2, 1e-1])
                theta = np.random.choice([0.1, 0.5, 1.0])
                #noise_generators = {brain.name: make_noise_generator(brain.action_size, scale, dt, theta) for brain in brain_spec_list}
                noise_generators = dict(map(lambda brain: (brain.name, lambda : torch.from_numpy(np.random.normal(loc=0.0, scale=scale, size=(brain.action_size))).float()), brain_spec_list))
                #noise_generators = {brain.name: lambda : torch.from_numpy(np.random.normal(loc=0.0, scale=scale, size=(brain.action_size))).float() for brain in brain_spec_list}
                learning_episodes += 1
            else:
                scale = 0.0
                noise_generators = no_noise

            steps_this_episode = 0

            while True:
                states = env_wrapper.get_states()
                print(noise_generators)
                actions = agent.act(
                    states, policy_suppression=(1.0 - scale), noise_func=noise_generators
                )

                # print(actions)
                # print(states)
                # Actions is now returned as {'brain1': [actions], 'brain2': [actions]}
                # The order in which states and actions are concatenated here, then together
                # determines the interpretation of the input neurons on the global critics
                # It has to match the states and actions that have already happened
                env_wrapper.step(vector_action=actions)

                if not online(i):
                    (
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                    ) = agent.replay_buffer.add_from_dicts(*env_wrapper.sars())
                    agent.learn(0.99)

                rewards = agent.replay_buffer.latest_reward
                dones = agent.replay_buffer.latest_done
                scores = [
                    scores[x] + rewards[x]
                    for x in range(agent_inventory.num_agents)
                ]

                steps_this_episode += 1
                print(f"Steps taken this episode: {steps_this_episode}")
                print(f"Dones = {dones}")
                if np.any(dones):
                    break
                if steps_this_episode > 500:
                    break

            if not online(i):
                episode_scores.append(max(scores))
                average_scores.append(np.mean(episode_scores))
            else:
                on_policy_scores.append(scores)
                on_policy_averages.append(np.mean(on_policy_scores))

            print("Total score this episode: {}".format(rewards))
        except (Exception, KeyboardInterrupt) as event:
            if isinstance(event, KeyboardInterrupt):
                print("Interrupted training, will save weights now")
            else:
                print("Encountered an unexpected exception, trying to save weights")
                exception_raised = event
            break

    results_directory = get_next_results_directory()
    agent.save_weights(results_directory)
    np.savetxt(os.path.join(results_directory, "scores.txt"), average_scores)
    np.savetxt(
        os.path.join(results_directory, "on_policy_scores.txt"), on_policy_averages
    )
    if note is not None:
        with open(os.path.join(results_directory, "note.txt"), "w") as f:
            f.write(note)
    params = {
        #"noise_fn": repr(noise_fn),
        "noise_fn_taper": noise_fn_taper,
        "taper": "linear",
    }
    with open(os.path.join(results_directory, "params.yml"), "w") as f:
        yaml.dump(params, f)
    print(f"INFO: Saved results to {results_directory}")
    if exception_raised is not None:
        raise exception_raised


@cli.command()
@click.option(
    "--weights-path",
    type=str,
    default=None,
    help="""
    Path to the directory containing the trained weights of the agents networks.
    Can be none, in which case the pre-trained weights in resources are used.
    """,
)
@click.option(
    "--n-episodes",
    type=int,
    default=1,
    help="""
    Number of episodes to train an agent for.
    """,
)
def run(weights_path: str, n_episodes: int):
    """Initialise an agent using pre-trained network weights and observe the
    agent's interaction with the environment.
    """
    env, brain_spec = get_unity_env(
        os.path.join(
            os.environ["PROJECT_HOME"],
            "./unity_environments/Tennis/Tennis_Linux/Tennis.x86_64",
        )
    )

    brain_spec_list = list(brain_spec.values())
    print(env.brain_names)

    env_wrapper = UnityEnvWrapper(env, brain_spec_list, ActionType.CONTINUOUS)

    # This defines an important ordering which sets which indices of the policy
    # and critic networks correspond to the observations and actions of each agent.
    # This list is passed to the agent inventory, which assigns slices to agent specs
    # and also is passed to the buffer.observe_environment method.

    agent_inventory = AgentInventory(brain_spec_list)
    agent = MADDPGAgent2(agent_inventory, hidden_layer_size=256)

    if weights_path is None:
        weights_path = path_from_project_home("./resources/solved_weights/")
    agent.load_weights(weights_path)

    for i in range(n_episodes):
        print(f"Episode: {i}")
        env_wrapper.reset(train_mode=False)

        scores = [0.0 for x in range(agent_inventory.num_agents)]
        for i in range(500):
            states = env_wrapper.get_states()
            actions = agent.act(states, policy_suppression=1.0)

            env_wrapper.step(vector_action=actions)

            rewards = env_wrapper.get_rewards()
            dones = env_wrapper.get_dones()
            scores = [
                scores[x] + rewards["TennisBrain"][x]
                for x in range(agent_inventory.num_agents)
            ]
            dones = [dones["TennisBrain"][x] for x in range(agent_inventory.num_agents)]
            if np.any(dones):
                break
        print(
            "Total score (averaged over agents) this episode: {}".format(
                np.mean(scores)
            )
        )


if __name__ == "__main__":
    os.environ["PROJECT_HOME"] = os.path.dirname(__file__)
    cli()
