# DDPG project

Task: implement the DDPG algorithm in TensorFlow. Test on an OpenAI Gym environment.

Rules:

- No peeking at any code that implements DDPG or a similar algorithm
    - Exception: stuck for more than 10 hours AND asked a friend
- As far as I can think of, anything else should be allowed, but in particular:
    - Original paper ("source material")
    - OpenAI Gym docs
    - TensorFlow docs

## 2019.04.20

Starting with actor-critic function.

- We need to make some assumptions about the API. For example, action space. But we can leave it for now.
- What is the shape of the input to the MLP?
    - We have `x` as [batch_size, obs_dim] and `a` as [batch_size, act_dim]
    - We would want to concatenate on axis 0, because that is the dimension that matches in size.
    - Correction: we want to concatenate _along_ axis 1, because dimension 0 matches in size.
- I am multiplying the output of the policy network by `act_limit` because that's what I remember. But I don't know why this is important, unless the output of the network is limited to [0, 1] or [-1, 1]. That could be the case - the `output_activation` argument could well be a sigmoid or tanh. Yeah, I reckon it's most likely to be tanh, so I'll go with that now.
- I have tested the `mlp_actor_critic` function with a MWE that prints outputs from random inputs.

## 2019.04.24

Implementing the core computation graph

- It will be helpful to look at the source material

## 2019.04.25

Implementing the core computation graph

## 2019.04.27

Implementing the core computation graph

- What is the reward dimension? Can we assume 1?
    - In general, reward can be a vector of any dimension
    - Does gym specify this?
- The policy gradient in the paper is not quite familiar...why is it grad(Q) _times_ grad(pi)?
- Not sure how to efficiently perform this polyak averaging with network parameters in TF

## 2019.04.29

Working out polyak averaging with network parameters in TF

- https://stackoverflow.com/questions/36193553/get-the-value-of-some-weights-in-a-model-trained-by-tensorflow

```python
# Desired variable is called "tower_2/filter:0".
var = [v for v in tf.trainable_variables() if v.name == "tower_2/filter:0"][0]

[v_targ for v_targ in tf.trainable_variables() if 'target' in v.name]
```

- Running through what I have so far
    - Reward is a scalar so I can't get a dimension from it. I expect in the context of `gym` reward will always be scalar, so let's drop the generality.
    - Trying to do this bulk-placeholder creation function is causing trouble. For now, I will keep it simple (though tedious) and make all the placeholders individually.
    - Interesting. Cartpole is discrete control so the action space does not have a `high` parameter. So I have switched to `HalfCheetah-v2`. Pretty confident `spinningup` uses it. Oh yeah, I think it's because DDPG is only for continuous control! I wonder how it would go as a discrete action classifier though...
    - Now I have a discrepancy in the dimensions with mlp. `act_dim` is only 1 but the action space has 6 dimensions. Why?
        - Because I am setting `act_dim` as the length of shape when it should be first number in the shape.
- Ok, I have an attempt at the polyak averaging that makes syntactic sense to me, and runs without error, but I'm still 50/50 that it's a correct implementation.
- Now would be a good time to implement a basic test of what we have, checking that polyak averaging works by tracking one or more network parameter values before and after an arbitrary (but known) update. Actually, given they are independently, randomly initialised, it may suffice to just run a Session without adding any more operations.

## 2019.05.01

Testing polyak average implementation

- We want to track the network variables (at least one) to check that the operation is performed correctly
- Ok, I've implemented a test
    - I assumed `tf.assign` updates the value in the first argument automatically, but also returns the updated value for convenience. This is my interpretation of the docstring. However, only when I assigned the return value to `targ_vars` did my test pass.
    - Read SO #34220532
    - Change:

        ```
        tf.assign(targ_vars[i], polyak * ac_vars[i] + (1 - polyak) * targ_vars[i])
        ```

        to

        ```
        targ_vars[i] = targ_vars[i].assign(polyak * ac_vars[i] + (1 - polyak) * targ_vars[i])
        ```

    - Necessary or not, it passes and should be compatible with the rest of it.

Action sample function

- Time to add a random seed option
- Step to replicate

    > Select action $a_t = \mu(s_t|\theta^{\mu} + \mathcal{N}_t$ according to the current policy and exploration noise

- Right so what is $\mathcal{N}$? A random process...Gaussian or uniform? I think Gaussian. Let's check the paper. 

    > For the exploration noise process we used temporally correlated noise in order to explore well in physical environments that have momentum. We used an Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) with [theta] = 0.15 and [sigma] = 0.2. The Ornstein-Uhlenbeck process models the velocity of a Brownian article with friction, which results in temporally correlated values centered around 0.

- Gosh OK. I'll stick with Gaussian at least initially.
- Temporally correlated noise though, that's interesting. So I imagine that gives it a sort of momentum in itself. It's random but it doesn't jump around erratically over time - the current value is correlated with the previous value. But how does a stochastic process with a sort of momentum help with exploration in the context of physical momentum?
    - Ok suppose the action is a torque, and the noise term is particularly large in magnitude at this moment. You don't want that to suddenly go near zero or opposite direction in the next time step, because that would muck up your flow. You want to explore, but _gradually_, allowing chunks of behaviour to run their course.
    - Now I feel like I get it, so I hope that explanation is reasonable!
    - Now I _do_ want to try this process! Pray to Python...
- 

Random thought: it would be a nice extra project, after this, to implement DDPG in the more recent TensorFlow paradigm, i.e. Keras and eager execution.

## 2019.05.02

Implementing action sample function

- Ok, I initialise the random process near the top. Then I have this

    ```
    def select_action():
        return pi + process.sample()
    ```

- Feeling like there's more to it...
- I mean, that is pretty much a direct translation of what the pseudocode says. The question is whether it gels with TensorFlow.
    - `pi` is a `tf.Tensor`
    - `process.sample()` is a `numpy` array
    - Right, so adding these two objects is probably not gonna happen.
    - So should we change the random process code to be all TF? Or find a roundabout way of making the addition compatible?
    - `tf.convert_to_tensor`?
        - How inefficient is it to do every time?
    - There was almost certaintly more to it in spinning up, but they weren't doing this random process API that I've made.
        - Oh wait, I might be thinking of TD3. That might have a more sophisticated action sampler than this.
    - Let's roll with it for now.
- Next thing is to implement the training loop (at least a skeleton of it).

## 2019.05.03

Implementing training loop

- Start by iterating over episodes
    - Should number of episodes be a function argument?
    - I think number of epochs is an argument, at least.
    - Then there is iterations or episodes per epoch. Which is which/are any one and the same?
    - I feel like epoch = episode is reasonable. An epoch in supervised learning on samples is a run through the entire set of samples. An episode is a run through the environment via a trajectory. Granted a trajectory is one specific sequence of possibly infinite, but there is some similarity.
        - But I can also see how choosing a certain number of episodes per epoch is gives a sufficiently broad range of trajectories on average, and is closer to the idea of an epoch in SL.
    - Pretty sure iterations <=> time steps. 
        - But ARE you sure?
        - From memory, iterations were ~10^4 in the spinning up plots. The paper reports most problems were solved in fewer than 2.5 million time steps. Many were solved in far fewer, but two orders of magnitude fewer? Look, I don't know. I'm not calibrated enough with deep RL. But it seems unlikely.
    - But does this matter right now? Let's run with a hypothesis and iterate.
- `steps_per_epoch` feels familiar
    - Every `steps_per_epoch` we report stats
    - Keep the episode-step nested loops, but continue to increment `steps` across episodes
    - Or, perhaps we set the episode to terminate if time step reaches `steps_per_epoch`, and thus there is one epoch per episode? That fits nicely, but still not confident. I mean, there would still be uneven steps per epoch if it can terminate before the limit.
- For now I will stick closer to the paper and use 'number of episodes' and 'max steps per episode' rather than epochs. Hopefully it will become clearer as I progress.
- Ah, I forgot about optimisers. Adam or...?
    - Yep, Adam in the paper.
    - Actor LR: 1e-4
    - Critic LR: 1e-3
    - Weight decay for critic: 1e-2
- I quickly implemented a buffer as a list of tuples, but now I think there may be more convenient formats to group the states/actions/rewards in batches as arrays.
    - Replay buffer size in paper: 1e6
    - It sounds like `np.random.choice` returns an `ndarray` even if the input is only array-like, so that's convenient
    - Tried doing it as a numpy array, but ran into trouble assigning an array of different-sized arrays to an index of...an array. Error is `setting an array element with a sequence.`
        - In light of this, is it easier to use separate but aligned buffers for each kind of variable?
    - My solution when using list of tuples:

        ```python
                # Store transition in buffer
        if len(buffer) < max_buffer_size:
            buffer.append((o, a, r, o2))
        else:
            buffer[t % max_buffer_size] = (o, a, r, o2)
        # Sample a random minibatch of transitions from buffer
        transitions = []
        for _ in range(batch_size):
            transitions.append(random.choice(buffer))
        x_batch = np.array([transition[0] for transition in transitions])
        a_batch = np.array([transition[1] for transition in transitions])
        r_batch = np.array([transition[2] for transition in transitions])
        x2_batch = np.array([transition[3] for transition in transitions])
        ```

        - The assignments at the end are tedious.

Side notes

- Paper does not include actions in Q until second layer

Next

- Set up optimisers
- Implement graph execution and backpropagation
