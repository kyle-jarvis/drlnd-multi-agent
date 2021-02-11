"""
Multi agent projects from Udacity's DRLND
"""

from unityagents import UnityEnvironment


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
    Number of episodes to train for
    """)
@click.option("--note", type=str, default=None, 
    help="""
    Note to record to .txt file when results are saved.
    """)
def train(n_episodes, note):
    """Train an agent in the 'one_agent' environment using DDPG.
    """
    pass

@cli.command()
def run(weights_path: str, n_episodes: int):
    """Initialise an agent using pre-trained network weights and observe the 
    agent's interaction with the environment.
    """
    pass


if __name__ == "__main__":
    pass