# Udacity DRLND
## P3 Multi agent / collaboration

### Environment

Two agents take actions in a tennis-style environment. Each agent controls the movement
of one 'raquet'. Each agent receives its own perspective of the environment.
As described in [here](./UdacityP3MultiAgent.md), agents receive a reward of -0.01
if the ball lands on the ground in their half of the court. Agents receive a reward
of +0.1 for 'batting' the ball over the net. Therefore, agents receive a high reward
by establishing volleys collaboratively. 

## Learning Algorithm

The learning algorithm used to collect the below results is the Multi-Agent Deep 
Deterministic Policy Gradients algorithm described [here](https://arxiv.org/pdf/1706.02275.pdf).
Each agent _i_ posses policy networks μ<sub>_i_</sub>, μ'<sub>_i_</sub>, and
critic networks _Q_<sub>_i_</sub>, _Q_'<sub>_i_</sub>. Critic networks are 'central'
in that they receive information from all agents' perspectives, as well as all
agents' actions. Policy networks act only on information local to the agent to 
which they belong. 

This allows for stable learning of policies that allow agents to behave collaboratively
or competitatively whilst only requiring local observations at run-time.


For the results illustrated below, the architecture was as follows:

A. Each agent is identical in its perception of the representation of the environment
and the action space that it can explore. Each agent has its own set of networks 
that are identical in their architecture.

B. Policy networks
1. An input layer of (24) neurons.
2. A first hidden layer of 128 neurons.
3. A second hidden layer of 128 neurons.
4. An output layer of 2 neurons.

C. Critic
1. An input layer of (24*2) + (4*2) neurons.
2. A first hidden layer of 128 neurons.
3. A second hidden layer of 128 neurons.
4. An output layer of 1 neurons.

All layers are fully-connected, with ReLU activations applied to the neurons, except
for the output layers, where the Actor network has a tanh activation applied and
the critic has a simple linear activation.

Training proceeds in the usual way. During training, noise can be added to the agents
action selection. The noise scale is tapered linearly over the course of training.
The output layers of the actor and critic networks are intialised with very small 
weights so that the noise is allowed to intially dominate the action selection,
and the critic outputs are initially small compared to the environment rewards.

The other layers in the actor/critic networks have their weights and biases intialised according to the Glorot/Xavier-Uniform [method](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_).

### Hyperparameters

1. Learning rate : 5e-5

2. Tau (soft update parameter) : 1e-3

3. Minibatch size : 256

4. Memory buffer size : 5e6

5. Gamma : 0.99

6. Noise distribution ([Ornstein Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)):
    - θ : RandomChoice[0.1, 0.5, 1.0] each episode.
    - σ : 1.0 → 0.0 over 10000 episodes.
    - μ : 0.0
    - dt : RandomChoice[0.01, 0.05, 0.1] each episode.

## Results

Initially, positive rewards are received very rarely. To ease the learning process,
the memory buffer size and minibatch size were increased from the values used
in earlier projects to solve comparable environments. The rate at which the noise
was tapered was also lowered. Ultimately, this gave rise to a very stable learning
trajectory, which rises to a value many times the solved threshold as the noise
tapers to a managable level.

<img src = "./resources/results.png" width="300"/>

The GIF below shows the agents, trained using MADDPG, playing a game in the environment.
As described elsewhere, agents receive a positive reward for hitting the ball
over the net, and a negative reward for 'dropping' the ball. Thus, they are encouraged
to establish rallies.

Whilst training, the episode length became so long due to the length of the rallies
that training was terminated prematurely.

<img src = "./resources/reacher.gif" width="300" height=180/>

## Future work

There is ample scope to incorporate many of the improvements and techniques
commonly reported in the literature to this 'vanilla' implementation. Examples include
batch normalisation and prioritised replay.

The agent rewards could also be trivially manipulated without requiring knowledge
of the underlying environment, and agent's perception, by rewarding the opposite agent,
say B, rather than penalising the current agent, say A, when the ball is dropped
in A's half, thus encouraging competitive play, which would be interesting to 
observe.