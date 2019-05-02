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
