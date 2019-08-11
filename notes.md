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

## 2019.05.05

Setting up optimisers

- `tf.train.AdamOptimizer`
    - Assume default betas
- With weight decay: `tf.train.AdamWOptimizer`
    - TF doc: 
        > It computes the update step of train.AdamOptimizer and additionally decays the variable. Note that this is different from adding L2 regularization on the variables to the loss: it regularizes variables with large gradients more than L2 regularization would, which was shown to yield better training loss and generalization error in the paper above
    - DDPG paper:
        > For Q we included L2 weight decay of 1e−2
    - Hmm
    - So if I use TF default weight decay, it may not be precisely the same. I am ok with this - I'm not trying for a perfect replication, I'm just trying to get it to work and be able to compare my results roughly to baselines.
- I was wondering if the optimisers apply to both main and target networks, but if targets are merely derived from a polyak average of main, then of course.

Optimiser updates

- Ok I'm looking at this policy gradient equation again. I was confused before. I think I get it.
    - You have grad(Q)*grad(pi). This is just the chain rule because (crucially) pi is substituted for the action in grad(Q)
    - Does this mean `pi_loss` should use `q_pi` not `q`? I'd say so. Otherwise `pi_loss` does not depend on pi! Remember this is an off-policy algorithm.
- Use `opt.minimize()` to compute AND apply gradients
    - Do I need to store the returned operation? I guess so. It could be one of the operations in `sess.run()`.

Target update

- I'm not sure how this is going to work. I have already `assign`ed the target update, do I also need to `sess.run()` it?
    - Think of the computation graph.
    - Ok I've drawn what I think the graph is. I'll get to the problem of the first iteration, but for now let's assume we are more than one iteration into it. Imagine we
        - Initialise the placeholder values
        - Feed through to `pi_loss` and `q_loss`
        - Update `ac_vars` by propagating the loss gradient back.
    - I would then expect the operation we created by assigning the polyak-average to target vars would also execute, using the updated `ac_vars`
    - Let's imagine the process with some personification.
        - Ok executing `critic_minimize` -> `q_loss` -> (`q`, `backup`)
        - `q` -> (`x_ph`, `a_ph`, theta_q)
        - `backup` -> (`q_pi_targ`, `r_ph`, `d_ph`)
        - `q_pi_targ` -> (`pi_targ`, `x2_ph`, theta_q_targ)
        - `pi_targ` -> (`x2_ph`, theta_pi_targ)
        - Now theta_q_targ has an assignment to it, but that operation was added after all this. So it gets values from `global_variables_initialiser`. Similarly for theta_pi_targ.
        - Feed forward!
        - Feed backward!
        - `theta`'s are now updated.
        - Next iteration
        - [Repeat graph dependency process above]. Oh, theta_q_targ depends on theta_q and the previous theta_q_targ? Ok, noted. Similarly for theta_pi_targ.
        - Feed forward!
        - ...
    - So now I feel like I get it, and there is no need to execute some target update process in the training loop. It's all in the graph.
- A couple of questions are raised from above thoughts
    - _Where_ should we place the target update assignment in code?
        - Should the `minimize` calls go before it? This follows the logic that TF would only execute intermediate operations if they were added to the graph before the top-level operation (I do not know if this is the case).
        - I imagine TF builds a tree of dependent operations. If we make the polyak assignment before the minimize operation, then it follows that the polyak assignment will execute and feed into the loss the first time. Given everything is kinda random initially, this doesn't seem like it matters much anyway (famous last words?). But best to be as correct as we think we can be.
    - For the first iteration, should the target variables be updated before feeding in to everything else?
        - The paper suggests not, because Algorithm 1 initialises the target parameters and then only updates them at the end of the training loop.
- In light of this thinking, I will move the `minimize` calls out of the training loop, before the target update assignment. Similarly the optimiser inits need to be moved up.

Next

- Read over the code vigilantly for any blunders
- Consider `step` function just being one iteration rather than a for loop
- Add basic stat reporting to training loop (so we can start testing!)

## 2019.05.08

Reading over code

- Default `hidden_sizes` was wrong way around
- The use of `'dense'` to get network parameters is not general (e.g. learning from pixels). Is there any general method?
    - Maybe I don't need `'dense'` at all. I put it there to avoid bundling in the higher level variables like `q` and `out`. But do these come under `tf.trainable_variables`? Probably not, because they are just outputs. Only the parameters are trainable.
    - We can test this out.
    - Can confirm that `'dense'` is currently unnecessary
- Ah, it was feeding in the tuple from the current time step $t$ instead of the replay buffer batch $i$.
    - Oh, I think this means `done` also needs to be stored in the replay buffer. That's a departure from the pseudocode.

Training loop position

- Currently we have the episode loop top-level in `ddpg()`, and the iteration loop wrapped in a `step()` function which is called each episode iteration.
- At the top of `step()` I initialise the environment with the initial observation, `o = env.reset()`. `o` is referenced for the buffer and reassigned with the next observation. If we moved the step loop outside `step()` so it was just one step, then I would have to parse in `o` and return `o` for reassignment.
- At least for now, `step()` isn't of benefit - the abstraction is only in syntax so that we have a simple-looking main loop.
- I'll keep it as-is for now

Reporting

- Data
    - Episode return
        - I can't see a way to get this directly, so we'll need to add it up ourselves
    - `q_loss`
    - `pi_loss`
    - Episode
    - Total steps
    - Steps per episode
- Statistics
    - mean, std, min, max
    - sum (for steps and return)
- API
    - Class seems best
    - We could carry a reporter object around with the `step()` function. Each step we let the reporter know the reward and it updates its internal value of return. Maybe q_loss, pi_loss too, but not sure if it is more informative to average that over the whole episode, or the last n steps, or the last step.
    - At the end of an episode we let the reporter know the final time step `t` so it can accumulate total steps, and steps-per-episode stats
- We also want logging, and from that, plotting...uh oh, scope creep?
    - It would make life easier to use the `spinningup` API here. But I would have to avoid any view of `ddpg.py`, and any calls to it (because I don't even want to see the function arguments).
    - It seems safe and compliant to import `spinup.utils.logx` and only interact with that.

## 2019.05.10

Configuring loggers

- Right, logger should start at the top of the function
- Can use `locals()` to collect all local variables in dictionary. At the top of a function, this will just be the input arguments.
- Also use `setup_tf_saver` after computation graph is set but before training
- Now to log during training
    - End of step: e.g. `epoch_logger.store(QLoss=q_loss)`
    - End of epoch: e.g. `epoch_logger.log_tabular('QLoss')`
    - End of epoch, to save to fild: `epoch_logger.dump_tabular()`
- Note `EpochLogger.log_tabular()` empties the value for the given key once logged. This makes sense to reset for the next epoch.
- What should I put in `state_dict` for `Logger.save_state()`?
    - > usually just a copy of the environment---and the most recent parameters for the model you previously set up saving for with setup_tf_saver
    - Most general would be `locals()`
    - "copy of the environment": `sess`?
    - Model parameters: `ac_vars`, `targ_vars`

Testing loggers

- Let's not worry about the algorithm performing! Just check the logger functionality.
- `save_config` is OK
- After fixing bugs below, ran 10 episodes (yeah, return stayed in the negative 100s...) and it all looks OK, except for `save_state`: `Warning: could not pickle state_dict.`
    - Currently `state_dict` is `{'trainable_variables': tf.trainable_variables()}`

Of course, we _do_ have to deal with any critical bugs first

- In `select_action`, `pi` is float32 while the noise is float64
- Ah, let's commit just to separate any bug fixes.
- `o2, r, done, _ = env.step(a)` throws `ValueError: setting an array element with a sequence`
    - If I just go `env.step(a)` without assignment in the debug console it gives the same error
    - From `env.step` doc: `a` should be "an action provided by the environment"
    - We are giving it a `tf.Tensor`, so that's probably the issue
    - What to do then?
    - If I follow what the doc is saying and look at `env.action_space.sample()`, it is a numpy array - so maybe we just need to convert the action to numpy by `sess.run`?
    - Like so: `a = sess.run(select_action(), feed_dict={x_ph: o})`
        - Shape issues
    - Alternative: run `pi` and then take away `tf.convert_to_tensor` on noise
        - Had to reshape `o` to have batch dimension (of 1 in this case), then `np.squeeze` to undo it because we add the batch dimension for the actual batch later

Next

- The hard phase!
- Worth checking over everything again
- Add time-stamp directory to logging path so we don't overwrite
- Find the simplest compatible environment
- Test with at least three different random seeds

## 2019.05.15

Read-through

- Add built-in `random` seed setting
- Using `a.shape[1]` in `mlp_actor_critic` is actually a TF object: `Dimension(6)`. So combining this with `hidden_sizes` gives [400, 300, Dimension(6)]. I wonder if this is a problem? The output shape is OK.
    - Switching to `action_space.shape[0]` since it seems safer
- Ohhhohohoho: `backup = tf.stop_gradient(r_ph + d_ph * discount * q_pi_targ)`
    - We want future value to go to 0 when we are done.
    - `d_ph` is True when we are done. Opposite!
    - `(1 - d_ph)` is what we want here. Crucial.
- Do we need an axis on these?

    ```
    q_loss = tf.reduce_mean((backup - q)**2)
    pi_loss = -tf.reduce_mean(q_pi)
    ```

    - No: Q is 1D and we want to average over the batch
    - Would it be more generally correct to specify an axis? I don't know. If we are only talking about Q then I think it should always be 1D for most (all?) SOTA algorithms today. Anyway, I'm not trying to be as general/abstract as possible in this implementation.
- Hmm, so I know there were decent reasons to make `buffer` a Python list but the appending might be fairly slow. Then again, if all these Tensors are just stored using references, maybe not. Besides, I doubt it is a major time sink in the overall algorithm.
- I think `critic_minimize, actor_minimize` should be added to the logger TF saver, but this shouldn't affect the algorithm
- Do we sample from the buffer with replacement?
    - How much difference would it make to have a small probability of duplicate samples?
    - If we are using the built-in `random.choice` we can only draw one at a time anyway
- It's worth checking the process noise magnitude relative to the action - this might vary significantly from problem to problem
- Move log calls before break condition in episode loop (we want to record loss on the last time step too)

Adding time-stamp directory to logging path

Test runs

- Just ran a couple on seed=2, and they give totally different results. So seeds don't seem to be working.

Test environments

- I think `HalfCheetah-v2` is considered fairly difficult.
- We need continuous control
- What are the easiest continuous control environments for DDPG?
    - `MountainCarContinuous-v0`?
        - May actually be pretty hard - has limited time steps to get it right
    - `Pendulum-v0`
        - It's simple and fast to run
        - The more rewarding states are less and less stable, so that's an argument for difficulty
        - Reward is sensitive, dynamics are fast

Next

- Continue scoping difficulty of environments for initial testing

## 2019.05.17

Environment for testing

- Ok, what we're going to do is run spinningup DDPG from terminal on some environments and get a sense of relative performance
    - `python -m spinup.run ddpg --env_name HalfCheetah-v2 --seed 0 10 20`
        - Erm, what is the stopping condition? It's gone 35 epochs on the first seed at time of writing. Don't tell me it's going up to 3e6...
    - Insights
        - It takes a long time to git gud. The jump from large negative to large positive return is relatively quick (e.g. over two epochs), and takes e.g. 8 epochs
        - We should have a test procedure
        - Once the return has scaled up to a large positive value (e.g. 2e3), standard deviation on return reduces a lot over time (seed 0), e.g. ~1e3 to ~1e2 or ~1e1.
- Useful page https://spinningup.openai.com/en/latest/spinningup/bench.html
    - It seems better to try an environment where DDPG has less variance and a higher lower-bound on performance
    - Benchmark qualities
        - HalfCheetah: fairly steady, high lower bound
        - Hopper: a bit volatile, moderate lower bound
        - Walker: a bit volatile, moderate lower bound
        - Swimmer: fairly steady, high lower bound
        - Ant: a bit volatile, low lower bound
        - Ok seems like HalfCheetah is relatively OK in the continuous control space.
    - Exemplar-ness
        - We need the test procedure for on-policy evaluation. Gosh, OF COURSE. Training is on a random batch from the buffer, it could be really old, bad actions!
        - 10 random seeds
        - Test every 10,000 steps 10 times on noise-free, deterministic, current policy
        - Batch size 100
        - I am skeptical of the paper's buffer size of 1 million if the benchmarks run ~million iteracts. But I'm not sure. Maybe the benefit of off-policy interactions increases without bound, and the buffer size is just a practical measure?
    - DDPG HalfCheetah benchmark has std of about 2000 in converged region. So given my current test run has std go down to ~1e2, the source of this high std seems to be the between-seeds variance.

Next

- Implement test procedure

## 2019.05.19

Implementing test procedure

- Ok, let's get a total step count happening
- Changing to a epoch-by-steps configuration instead of episodes
    - The implication seems to be that we don't quit when `done`, and epochs generally run unevenly over episodes. What are the implications of this? Assumption of infinite horizon? No, there is a discount
    - Bug: first episode is 1000 steps long, remaining episodes are 1 step
        - Ah, I need to do `o = env.reset()` in `done` condition too
- Onward
    - Huh. I kept the run that was verifying the new structure going up to 28 epochs, and as far as I could see it was all bad except one epoch (14) with return 385 to 691, 483 +- 110. Wow. I guess we have to take it as a fluke, or unstable training.
    - Anyway, I now have a test function.
- Testing it out
    - Logging works
    - Why would we run 10 trajectories (assuming =episodes) if it is a deterministic policy? Are the initial conditions of the environment randomised? The dynamics?
    - Again, random seeds don't seem to (at least solely) determine the results here.
        - Setting the `gym` random seed might fix it
    - Performance is bad, in fact it may have gotten worse on average, but I haven't plotted

Other

- Noticed exploration noise is a scalar - we need to specify the shape as that of the action
- Questioned `-tf.reduce_mean(q_pi)` vs. `-tf.reduce_mean(q)` but became confident again. The graph runs based on the placeholder feeds, and we are feeding in the buffer-sampled state $s_i$. The paper says in the policy loss, the policy action is substituted in Q.

Next

- We're at a really difficult stage now. It's hard to know the best place to start in making this work.
    - If that's the case, the best we can do is work through possibilities/verify functionality in turn.
- First of all just fix the seed and exploration noise issues. Then run 50 epochs on at least 3 seeds, and plot the results, so we can be confident whether it is otherwise working.
- I'm uncertain whether we should start really probing under-the-hood, or keep trying variants and additions to the hyperparameters, algorithm.
    - Hyperparameters seem least likely to cause trouble here. AFAIK they match the paper. They don't seem extreme.
    - The scale of the exploration noise maybe needs to be scaled by the action limits
    - Initial uniform random exploration could be important, but I don't think the paper mentions it. Based on that it seems useful but not crucial to get a positive result. But you never know, this is DRL.
    - Whatever we do, we gotta maintain that scientific mindset. What do I _observe_? What are all the possible explanations for the observation that I can think of? How likely are these?
- Things I'm unsure about
    - How terminal states affect objectives and learning
    - The importance of buffer size
    - The most sensible test procedure, and why trajectories would vary for a deterministic policy

## 2019.05.21

Setting `gym` random seed: `env.seed(seed)`

- Test (3 epochs, seed=0)
    - Run 1: 2019-05-21-20-28-44_ddpg_halfcheetah-v2_s0
    - Run 2: 2019-05-21-20-33-13_ddpg_halfcheetah-v2_s0
    - Nope (e.g. average return epoch 1 -474.3441 vs. -449.04657)
- seed=1234: nope
- Ok, I need to get serious. Based on some googling this is a fiddly procedure. Let's make a minimal test and build it up.
    - Separate script checking that simple lists/arrays replicate
    - `random`: check
    - `np.random`: check
    - `tf.random`: check
    - `gym`: check
    - Ok, so maybe the problem is doing random TF ops in different scopes/files
        - Same file, different scope first: check
        - Same file, function scope: check
        - As above, plus `tf.layers.dense` with `np.random.normal` fed input: check
        - Different file, else as above: check
- Back to main
    - First action in training, evaluated
        - Run 1: [-0.00724385  0.00492079  0.03382656  0.01570793 -0.00460604  0.00858583]
        - Run 2: [-0.00724385  0.00492079  0.03382656  0.01570793 -0.00460604  0.00858583]
        - Huh, interesting. So we have _some_ replication. Where does it diverge?
    - First `o2` in training, evaluated
        - Run 1: [-0.08694839  0.09207987  0.00348153  0.01589041  0.0017272 ...
        - Run 2: [-0.08694839  0.09207987  0.00348153  0.01589041  0.0017272 ...
    - Pick this up next time

Specifying shape of exploration noise

- Should the `np.random.randn()` in `sample` have a dimension?
    - I think so. It makes sense to have independent randomness on each component of the action.
- Should it be `(None, act_dim)` or `(act_dim,)`?
    - The latter, because we use it on a single action. But all the vector values are independent, so if we ever used it in a batch we could go `noise[0]`
- Test
    - Looks fine

Next

- Continue investigating where pseudorandomness diverges in our seed setup

## 2019.05.25

Investigating where pseudorandomness diverges in our seed setup

- Ok so last time
    - Testing whether the random seed gave consistent results
    - Created a separate file to test this, emulating some of the operations in the actual program
    - That seemed all good, so we moved to tests based on matching variable values in the actual program
    - Matches were found up to the assignment of `o2` in the training loop. We haven't found a case where the results become inconsistent (i.e. diverge) - we just know that they are inconsistent at the output statistics level.
- Follow-the-seed
    - Working seed: 1234
    - Let's test `o2` again
        - Run 1: `[-0.08694839,  0.09207987,  0.00348153,  0.01589041,  0.0017272 ...`
        - Run 2: `[-0.08694839,  0.09207987,  0.00348153,  0.01589041,  0.0017272 ...`
        - Yep
    - Let's compare `o2` in the final iteration (`step==steps_per_epoch-1`)
        - Run 1: `[-5.77279067e-01,  3.30225890e+00, -3.51571425e-01,  5.34152396e-01 ...`
        - Run 2: `[-5.77277104e-01,  3.30223145e+00, -3.55792556e-01,  5.28771697e-01 ...`
        - Run 3: `[-5.77279480e-01,  3.30226571e+00, -3.51553730e-01,  5.34153603e-01 ...`
        - Bingo
        - Interesting that the values are very close though
    - `o2` in second iteration (`step==1`)
        - Run 1: `[-0.10768196,  0.06230653, -0.02311073,  0.02713979,  0.0832682 ...`
        - Run 2: `[-0.10769117,  0.06231164, -0.02317908,  0.02720147,  0.08327151 ...`
        - Bingo
    - Hypothesis: divergence is cause by the parameters in the MLPs being initialised differently
    - Test: use `env.action_space.sample` for action instead of MLPs (this only tests the actor network)
        - `step==1`
            - Run 1: `[-1.16243351e-01, -1.27413451e-02, -2.81861979e-02,  4.07141069e-01 ...`
            - Run 2: `[-1.16243351e-01, -1.27413451e-02, -2.81861979e-02,  4.07141069e-01 ...`
            - Run 3: `[-1.16243351e-01, -1.27413451e-02, -2.81861979e-02,  4.07141069e-01 ...`
            - Same
        - `step==steps_per_epoch-1`
            - Run 1: `[-8.57854903e-02, -6.22277843e-02,  1.32189816e-03,  1.77853597e-02 ...`
            - Run 2: `[-8.57854903e-02, -6.22277843e-02,  1.32189816e-03,  1.77853597e-02 ...`
            - Same
    - Ok, the test does not narrow down enough yet. It does suggest that `gym` is not contributing to the problem.
    - To narrow down further, let's test the unnoisy action, i.e. removing `process.sample()` from `select_action()`
        - `step==steps_per_epoch-1`
            - Run 1: `[-5.77281139e-01,  3.30230450e+00, -5.13863239e-01,  5.02119414e-01 ...`
            - Run 2: `[-5.77275336e-01,  3.30221288e+00, -5.09267961e-01,  5.32657678e-01 ...`
            - Same
        - Ok, so `pi` is a problem.
- The evidence points to the MLP parameters being the source of divergence. This means I need to review how to ensure consistent initialisation/graph in TensorFlow.
    - SO#38469632 suggests setting an operation-level seed for each operation that involves randomness. Comments say this was not used in a TensorFlow example, and it may have been a past bug that is now fixed.
        - Another answer says you should completely restart the Python interpreter every run, or call `tf.reset_default_graph()` at the start.
    - I see this pattern around (e.g. Keras FAQ)
    
        ```
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
        tf.set_random_seed(1234)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        ```

        - The first line is to set single-threading because multi-thread is a potential source of non-reproducibility

    - Also this, as the first seed setting:

        ```
        import os
        os.environ['PYTHONHASHSEED']=str(seed_value)
        ```

      "This is necessary in Python 3.2.3 onwards to have reproducible behavior for certain hash-based operations (e.g., the item order in a set or a dict..." - Keras FAQ

        - I don't see how that would be problematic in our case...I don't think e.g. item order in a dict would determine these results
        - Keras FAQ doesn't use above code, instead suggesting `PYTHONHASHSEED=0` before `python` in the terminal command
- Trying out solutions
    - First, let's try `graph=tf.get_default_graph()` as an argument to `Session`
        - Run 1: `[-5.77274257e-01,  3.30220153e+00, -5.13445612e-01,  5.27330560e-01 ...]`
        - Run 2: `[-5.77276003e-01,  3.30221998e+00, -5.08975840e-01,  5.32709864e-01 ...]`
        - Different
    - Now, let's try using that `config` argument to the session
        - Run 1: `[-3.21516184e-01,  7.35947181e-01, -4.74953472e-01, -4.64066919e-01 ...]`
        - Run 2: `[-3.21516184e-01,  7.35947181e-01, -4.74953472e-01, -4.64066919e-01 ...]`
        - Run 3: `[-3.21516184e-01,  7.35947181e-01, -4.74953472e-01, -4.64066919e-01 ...]`
        - Stop debugger (rather then restart)
        - Run 4: `[-3.21516184e-01,  7.35947181e-01, -4.74953472e-01, -4.64066919e-01 ...]`
        - Restart computer
        - Run 5: `[-3.21516184e-01,  7.35947181e-01, -4.74953472e-01, -4.64066919e-01 ...]`
    - There you go!
- I am surprised that configuring the threads to single is the solution. It seems the MLP initialisation is consistent under these conditions after all.
- This raises the question: is it at all possible to get reliably reproducible results with multiple threads?
- Bringing back `process.sample()` in `select_action`
    - Run 1: `[-0.00701224,  1.33773652, -0.44480645, -0.42450498, ...]`
    - Run 2: `[-0.00701224,  1.33773652, -0.44480645, -0.42450498, ...]`
    - Same
- Checking for match on output results (i.e. `progress.txt`)
    - This is currently 2 epochs, 1000 steps per epoch
    - Match!
- Change seed from `1234` to `0`
    - Result totally differs from `1234`
    - Results match on multiple runs
- I am now confident that for any consistent random seed we get consistent results

Next

- Run the current implementation on 10 different random seeds, 50 epochs, to get a reliable sense of the current performance and the issues that may be present.

## 2019.05.27

Running the current implementation on 10 different random seeds, 50 epochs

## 2019.05.28

Reviewing test results

- Plot command (using Spinning Up again)

    ```
    python -m spinup.run plot ./out/test0 -x TotalSteps -y AverageReturn AverageTestReturn PiLoss QLoss
    ```

    - I modified SU's `plot.py` to use the stat names I use. In future I should probably just conform to their stat names.

- Observations
    - For reference: this is HalfCheetah-v2
    - Average train return is fairly flat. Slight overall upward trend from -500 to -400, but we can't be at all confident in this given the variance and low sample size (which remember, is 10).
    - Average test return is less flat than train. Slight overall upward trend from -600 to -400, but still not confident that this is meaningful. Judging by the variance shading, there is at least one case where it starts and ends at about the same return.
    - Test return has about the same overall variance between runs as train return.
    - Test return has a higher variance over time than train return. In lay terms, it is more spiky.
    - Average policy loss is relatively smooth, with a slight overall downward trend from 0.2 to 0.1. Variance between runs is relatively high (std about 0.2). In some cases loss goes negative.
        - I think it is invalid to interpret policy loss as a series. Each loss value is only meaningful at the time step is is computed, because the feedback loops between pi, the environment and q will change the meaning of the loss at each step.
    - Average Q loss is relatively smooth, with an overall upward trend from 0.02 to 0.06. Variance between runs is relatively high (std about 0.02) and grows over time.
        - I expected this to decrease over time. Q is trained on the mean-squared Bellman error. Then again, this depends on pi through `q_pi_targ`, and I don't know what could go wrong with pi.
        - Would it be feasible to isolate training of Q by using a random policy, instead of pi? Removing pi from the picture should make it easier to verify whether Q is training correctly.

Next

- Listing out all possible problems with the implementation

## 2019.05.29

Possible problems with the implementation

- Tensor dimensions are incorrect
- Loss equations are incorrectly implemented
    - The inputs that are fed in
    - Target vs. actor-critic variables
    - Use of `done` variable in backup
    - Mean operation (axes?)
- Target parameter updates incorrectly implemented
    - Construction for graph
    - Execution
- Target parameters are not initialised to actor-critic parameters
- Polyak parameter is too low (i.e. target networks track actor-critic too slowly)
- Experience buffer is not implemented correctly
- Experience buffer size is problematic
- Transitions are incorrectly assigned or not stored correctly in buffer
- Loss minimisation is not called correctly
    - Review a standard code example for this
    - I feel fairly confident that Q loss should trend down but we found it trends up. I'll give it a bit more thought and we can investigate this after the basic checks in this list.
- Exploration noise is too high or too low in magnitude
- Exploration noise is problematic in terms of its dynamics
- MLPs are not constructed properly
- Not executing all the operations that need to be executed in `sess.run`
- Environment interaction and management in training loop is incorrect

Known issues

- Once buffer reaches capacity, under the current use of a Python list we need to track the total number of steps and use that to index the buffer, rather than `step`. We could move `total_steps` to the inner training loop and increment it one-by-one.
- `process.reset()` (the exploration noise process) should be called each episode, not each epoch

Fixing buffer indexing

Moving and repeating `process.reset()`

- Running this on seed 0. Whereas the seed 0 in the 10-seed test run matches an earlier seed 0 run, we are now getting completely different results. This indicates the random process resetting is in effect.

Loss minimisation calls

- Seem to be in order

We could try visualising the computation graph in Tensorboard. That may help!

Next

- Visualise computation graph in Tensorboard
- Plan a setup to train the critic on its own (with random actions)

## 2019.05.30

Setting up computation graph visualisation

- Done. Nice!

Examining graph

- Hmm, this has made me think that I'm not doing target variable assignment properly
- For one, I don't think I am initialising the targets to the actor-critic weights
    - Added an op for that, similar to target update except without polyak
- Added explicit run calls for these target updates, i.e. `sess.run(targ_init)` and `sess.run(targ_update)`
    - Do I need a `feed_dict` here?
        - I assume no, because we ran `global_variables_initializer()`. The weights don't depend on the placeholder values. Then again, maybe they do in a very long-winded way, via the gradient updates!
    - Well, the code runs... watch this space
- Ok, testing out these new ops
    - Stats definitely look different
    - For example, PiLoss is generally higher in magnitude, ~10^0
        - It's also positive and increasing. The critic is harsh and getting more harsh!
    - We need to give this a more thorough look, compare it to the first 10-seed test

## 2019.06.01

Comparing logs before/after target variable op changes

- Q loss has increased an order of magnitude from 1e-2 to 1e-1. This cannot be explained by between-seed variance, since we are using one of the same seeds.
- New Q loss has an overall upward trend like before, but this run has a plateau between 10 and 30 thousand steps. However, with only one new run to go on, we can't draw a general conclusion about this.
- Pi loss has increased an order of magnitude from 1e-1 to 1e0. 
- Where before Pi loss was about flat, it is now steadily increasing.
- Average episode return looks similar
- Average test return now has less variance over time
- All in all, this is a bad smell to me. But I am not well calibrated on what the loss trends should look like. It may be that we _have_ fixed a problem, but in its absence, other problems are causing different but similarly incorrect behaviour. Still, it is important that we get strong verification that the target variables are initialising and updating correctly. We should be able to do this by inspecting the output of the `sess.run` calls.

Checking target op initialisation

- All seed 0 unless stated otherwise
- `sess.run(targ_init)`
    - 12 arrays -> `pi_targ` and `q_targ`, each with input,hidden1,output, each with weight,bias
    - What are presumably the bias arrays are all initalised to 0. exactly
    - Weight arrays look like a bunch of random
    - Need to check this matches the actor-critic vars, so will need to create an op and run for that as well
    - `targ_init_eval[0][0, :5]`: `[-0.05512697,  0.094685  , -0.06746653,  0.09940008, -0.01107508]`
- `sess.run(ac_vars)` before `sess.run(targ_init)`

    ```
    [np.all(targ_init_eval[i] - ac_init_eval[i] == 0) for i in range(len(targ_init_eval))]
    [True, True, True, True, True, True, True, True, True, True, True, True]
    ```

    - `targ_init_eval[0][0, :5]`: `[-0.05512697,  0.094685  , -0.06746653,  0.09940008, -0.01107508]`

- `sess.run(ac_vars)` after `sess.run(targ_init)`

    ```
    [np.all(targ_init_eval[i] - ac_init_eval[i] == 0) for i in range(len(targ_init_eval))]
    [True, True, True, True, True, True, True, True, True, True, True, True]
    ```

    - `targ_init_eval[0][0, :5]`: `[-0.05512697,  0.094685  , -0.06746653,  0.09940008, -0.01107508]`

- So initialisation seems to be OK.

Checking target op update

- This ran without error in the episode loop:

    ```python
    ac_eval = sess.run(ac_vars)
    if total_steps <= 1:
        targ_eval = sess.run(targ_init)
    expected_targ_eval = [polyak*ac_eval[i] + (1-polyak)*targ_eval[i] for i in range(len(targ_eval))]
    targ_eval = sess.run(targ_update)
    for i in range(len(targ_eval)):
        assert np.all(targ_eval[i] - expected_targ_eval[i] == 0)
    ```

- Also used above to check the actor critic network is updating in response to gradient (just updating at all - it would of course be very hard to verify that it is updating by the right amount).
    - Initial `ac_eval[0][0, :3]`  : `[-0.05522674,  0.09458509, -0.06746653]`
    - Initial `targ_eval[0][0, :3]`: `[-0.05522674,  0.09458509, -0.06746653]`
    - Next `ac_eval[0][0, :3]`     : `[-0.0553144 ,  0.09448526, -0.06754046]`
    - Next `targ_eval[0][0, :3]`   : `[-0.05522683,  0.09458499, -0.0674666 ]`
    - Yep, that checks out. Great!
- Ok, stripping away that test code, let's see if we have different target update values if we don't call `sess.run(targ_init)`:
    - Initial `ac_eval[0][0, :3]`: `[-0.05522672,  0.09478495, -0.06746653]`
    - Next `targ_eval[0][0, :3]` : `[-0.02119069, -0.04744682, -0.0929214 ]`
    - Right, totally different. So the `sess.run` call is necessary to make the target networks initialised as the actor-critic networks.

This session's results indicate that we are moving in the right direction, even though performance is just as bad or worse than before.

Next

- Understanding the signs of good Q network training
- Implementing isolated Q training (i.e. random actions, no policy)

## 2019.06.03

Understanding the signs of good Q network training

- Baseline (from back when doing SU exercise 2.2)

    ```
    python -m spinup.run plot ../spin/spinningup/data/2019-04-19_ex2-2_ddpg_halfcheetah-v2/2019-04-19_21-03-12-ex2-2_ddpg_halfcheetah-v2_s0 -x TotalEnvInteracts -y AverageEpRet AverageTestEpRet LossPi LossQ
    ```

- Yep, baseline got good return so it is a good example
- Q loss
    - Looks like it doesn't divide by batches because the values are about two orders of magnitude higher than mine. Given the batch size is constant, this is fine.
    - Well, well. Q loss trends up in the baseline. I was wrong.
        - But...why?? Why would Bellman error increase as the model improves? Does it flatten out if you train for longer?
    - So this is actually somewhat encouraging. Our Q network might be OK.
    - Looking at a different run (2019-04-19_22-23-52-ex2-2_ddpg_halfcheetah-v2_s0), the loss curve is in a U shape, but still trends upward overall.
        - Apparently the same random seed as previous, so it seems like SU uses seeds for variety and statistical power, but not for exact reproducibility. Maybe it is a good thing to not seed the environment? This may explain the variance in test return I have seen in baselines.
    - 2019-04-19_22-33-28-ex2-2_ddpg_halfcheetah-v2_s20 similar. The U shape seems to be the more common case.
- Pi loss
    - Mine consistently trends upward
    - Baselines tend to trend upward for the first 3-4 epochs, then down (true of all three seeds that I checked)
    - Baseline is consistently negative, mine is consistently positive

## 2019.06.04

Changing pi loss and seeing what happens

- Yep. I am just interested in how the behaviour changes, even if it seems wrong to me.
- 10 epochs, 5000 steps per epoch, seed 0
- Changing `reduce_mean` to `reduce_sum` (for Q as well)
    - Similar
- Changing `q_pi` to `q`
    - Wow. Big change folks. We are on.
    - Average return has gone from being in the realm of -500 to -75
    - Q loss is that U-shape, mostly upward
    - Pi loss is still strongly trending upward and positive
    - Let's run this for longer: 20 epochs
    - I think I understand why this is correct now.
        - In both cases, the gradient has to back-propagate through the Q-network. This is fine as long as it doesn't modify the Q network (something to think about).
        - Then, it obtains the gradients for the output of the pi network, which was the action input to Q.
        - In the case of `q`, these gradients are then backpropagated through the pi network, modifying the parameters, and we are done.
        - But in the case of `q_pi`, the gradients have to go through the nested pi network input, and then backpropagate through pi _again_ to adjust the parameters.
        - For analogy, think about GANs. You don't use D(G(x)) for the generator loss. Instead you evaluate the output g = G(x) and then use D(g).
    - The 20 epoch run still didn't go anywhere outside that -50 to -100 range, but given the above reasonining, I think this is a step in the right direction.
        - This new behaviour smells a bit like the incorrect tensor dimension bug, but from memory that gave returns hovering closer to zero. Still, I could have a variant of the bug. Watch this space.
- Un-negating
    - This is the biggest long shot
    - The return stats are about the same (with `q` not `q_pi`), but naturally, pi loss is now trending down and negative.
        - Which just seems silly. My understanding is that pi loss is only meaningful in the iteration it is applied, but still, a process that is trying to decrease a metric ends up consistently increasing it and vice versa? Really confusing.
    - We can't rule out further problems with the Bellman predictor that is Q.

Next

- Check tensor dimensions, especially those feeding into the losses

## 2019.06.06

Checking tensor dimensions

- I am satisfied with it after probing most of the variables in detail

Trying out initial exploration

- So I am using `env.action_space.sample()` if `total_steps` is less than a certain threshold, otherwise as before
- What threshold?
- Should I not train the networks? Only train the Q network?
- At the moment I am working on a different idea: accumulating the replay buffer to a decent size before doing any training. It seems like this wouldn't hurt matters...just starting off with a wider pool of experience before the actor-critic starts learning. I just feel that the high number of repeated samples when the buffer is near-empty could be problematic.
    - Then again, maybe this is by design: get the networks training on something consistent. This doesn't feel plausible to me though.
    - If I set the non-learning phase to `steps_per_epoch - 1`, such that we are choosing actions randomly, the return is -334 (on this particular run) compared to the -50 to -100 range we've been stuck on. This suggests that the model is doing _something_ right.
- Should one freely explore in every epoch, or just the first?

Next

- Review what we changed today because I am tired
- Try pure exploration every epoch. And for goodness sake, try a different seed...

## 2019.06.08

Reviewing changes

- Change `q_pi` to `q` for `pi_loss`
    - Hmm, I think this was earlier and should have been committed
- Add exploration steps
    - Set a fixed number of "exploration steps"
        - This is to introduce an "exploration phase" of training. I had two uses in mind: accumulate experience in the buffer before drawing on them, and give the critic a head start before evaluating the actor, such that it gives more accurate evaluation and greater stability.
    - Changes to the training loop until _global_ step reaches the number of exploration steps:
        - Choose the current action randomly, _not_ according to actor
            - Increases the variance in experience, to benefit the critic
        - Perform the gradient descent step on the critic and _not_ the actor
            - Gives the critic a head start
- Move advancing the observation (i.e. `o = o2`) and the `if done` procedure forward, before the network training ops
- Only run 1 test trajectory instead of 10
- Nothing stands out as a bad choice, but I think using _global_ step for the exploration phase might not make as much sense as using the epoch step.
- My rationale for the exploration and critic-head-start is based on recall of the baseline implementation, and a technique I have seen for GAN training. Pseudocode:

    ```
    for epoch in epochs:
        i = 0
        while i < number_of_data_points:
            discriminator.set_trainable(true)
            generator.set_trainable(false)
            j = 0
            while j < discriminator_iters and i < number_of_data_points:
                j += 1; i += 1
                do_discriminator_training_iteration()
            discriminator.set_trainable(false)
            generator.set_trainable(true)
            do_generator_training_iteration()
    ```

    - Here, the discriminator is trained `discriminator_iters`-times more than the generator
    - The extra training is done every epoch

Running no-exploration (but change of `q_pi` to `q` in `pi_loss`) for seeds 10, 20, 30, 20 epochs, 1 test trajectory per epoch

- Just because we ought to try different seeds sometimes!
- Oh rats. I think I should be putting `update=True` when calling `process.sample`, to update the state.
    - Nope, silly. `update=True` is the default and it's what we want!
- Looking at the training go here...definitely significant variance between seeds. Worth trying regularly.
    - Something might be off though. Both the train and test return for seed 10 and 20 are in the 0 to -20 range. This doesn't seem normal for the first epochs, too good to be true. Be vigilant.
    - Well, seed 30 is in a -80 to -140 range, so it could just be seed ideosyncrasies.
    - So seed 0, 30 are alike, seed 10, 20 are alike
- I'm thinking about why pi loss would go up
    - Exploration noise magnitude is too high, so it is not evaluated well to begin with. Critic becomes more accurate and the noisy action gets a worse review over time.
        - Yeahhhh...probably:

            ```
            process.sample()/a_pi
            array([ -7.7017942 ,   4.89124961, -12.97847115, -42.12927971,\n         0.94239252,  -8.46267434])
            process.sample()/a_pi
            array([ -7.98796825,   3.9495834 , -11.8698725 , -47.99462962,\n         0.86044746,  -8.18483581])
            process.sample()/a_pi
            array([ -7.61431421,   3.95606896, -11.58792034, -46.44039802,\n         1.00007891,  -8.97872908])
            process.sample()/a_pi
            array([ -7.85591755,   3.68207894, -11.56466114, -48.57379345,\n         0.99131507,  -8.06771304])
            ```

    - Something is off about the use of action limits. Maybe the wrong index or axis or dimensionality.

Next:

- Run test with reduced exploration noise
    - Time-independent random normal
    - Process

## 2019.06.10

Running test with reduced exploration noise

- `np.random.normal`
    - I think an std of somewhere between 0.001 and 0.1 makes sense. Let's try 0.01.
    - Seeds: 0, 10, 20
    - Hang on a tic...I'm looking at the results for the last seed=0 run I did, compared to this one. Training return in this one has a higher average (about -63 vs. -80), and much lower std (about 3 vs. 15). But test return is identical. You can't explain that.
        - Checked it is running actor_minimize, but based on `sess.run(ac_vars[0])` there is no change
        - I will let this run to completion then check the initial values
        - I mean, the networks not updating is a good explanation for the stagnant runs we have been observing for a while...
        - Yeah, I'm not seeing a change from the initialisation. I am worried that it's a scope issue. Let's commit and then test removing the training function scope.
        - `pi` is the same every time, `q` is changing, from step to step but with me keeping the inputs fixed
        - Actually...having discovered this problem, I am changing my mind about `q` vs. `q_pi` in `pi_loss`. The loss function needs to depend on `pi`'s output in some way! Maybe `q_pi` is incorrect, but I don't think `q` is correct either. Let's check...
            - Yep. Changing `q` back to `q_pi` makes `pi` change its output each time step (again keeping the input fixed).
        - I was wrong!
        - And this works fine within the function scope, so keep it.
    - Ok, now that the `pi` network is at least changing (correctly, who knows...), let's try this version of exploration noise again

Big news!

- Random normal noise std 0.1
- 1000 exploration steps per 5000-step epoch
- Seed 0 was a good run, return got consistently positive, usually 100s (!!!)
- Seed 10 didn't go anywhere (goes to show, always try different seeds)
- Seed 20 seems lucky and gets positive return after the first epoch

Next

- Remove the exploration steps as an ablation study, to make sure what the cause of this success is
- Try slightly lower e.g. half the current std of noise (it seems a little too high but we'll see)
- Revert to time-dependent noise process and play with parameter values

## 2019.06.11

Trying seed 0 without exploration steps but with Gaussian exploration noise

- Plot

    ```
    python -m spinup.run plot ./out/2019-06-11-13-43-49_ddpg_halfcheetah-v2_s0/ -x TotalEnvInteracts -y AverageEpRet AverageTestEpRet LossPi LossQ
    ```

    - It's a rollercoaster
    - I'll have to pick this up another day -- get a clearer picture on the differences between runs (0.1 vs. 0.05 std in noise, 0 vs. 1000 exploration steps)

Tried seed 0 (?), 10, 20 with Gaussian 0.05 std, no exploration steps

Tried seed 0 (?), 10, 20 with Gaussian 0.05 std, 1000 exploration steps

## 2019.06.12

Organising log files

- Before the next experiments let's change the logging system to handle experiment-name directories (i.e. not just time stamps), and add command line argument parsing
- Once we have a better idea of what the effects are of the current things I'm playing with, let's look into shortening the episode length, reducing the frequency of computed actions, and threads. The last one will compromise random seed reproducibility and should be left until we are more confident in how more consistent  performance can be achieved.
- Last run
    - Seed 10, 20; 1000 exploration steps; Gaussian exploration noise 0.05 std
    - 2019-06-11-(17-13-05,17-41-31)
- Second last run
    - Seed 0, 10, 20; 0 exploration steps; Gaussian exploration noise 0.05 std
    - 2019-06-11-(15-34-15,16-07-39,16-39-40)
- June 10th runs
    - Seed 0, 10, 20; 1000 exploration steps; Gaussian exploration noise 0.1 std
    - 2019-06-10-(21-44-02,22-12-40,22-40-51)
    - Earlier runs from June 10th are all seed 0 so presumably incomplete preliminary tests
- Ok... the first run I did on the 11th (2019-06-11-13-43-49_ddpg_halfcheetah-v2_s0) is seed 0, and that would correspond to 0 exploration steps, but it is unclear whether it was 0.1 or 0.05 std on Gaussian exploration noise.
    - Based on my to-do list I reckon noise was still at 0.1
    - This means we are missing the 1000-0.05 combo for seed 0
    - Let's run it and check if it matches any other runs - that will help verify
    - Next time, I will record all runs I am doing with exact parameters...

Running seed 0; 1000 exploration steps; Gaussian exploration noise 0.05 std

Running seed 10,20; 0 exploration steps; Gaussian exploration noise 0.1 std

Ok, now we have seeds 0, 10, 20 for 4 variants: (0, 1000) x (0.1, 0.05)

- Customising the SU plot script (copied to this repo) to add save functionality, because saving figures in MPL's GUI is awful
- Ok. There are basically three classes of performance here.
    - Bad (B): consistently in the -100s, around -500
    - Mixed (M): achieves good performance at multiple epochs, but fluctuates, sometimes going bad again
    - Good (G): gets good and stays good
- 0, 0.1
    - ![AverageEpRet](./out/test_exp_0_gauss_0p1/plot/AverageEpRet.png)
    - ![AverageTestEpRet](./out/test_exp_0_gauss_0p1/plot/AverageTestEpRet.png)
    - ![LossPi](./out/test_exp_0_gauss_0p1/plot/LossPi.png)
    - ![LossQ](./out/test_exp_0_gauss_0p1/plot/LossQ.png)
    - B=2, M=1, G=0
- 0, 0.05
    - ![AverageEpRet](./out/test_exp_0_gauss_0p05/plot/AverageEpRet.png)
    - ![AverageTestEpRet](./out/test_exp_0_gauss_0p05/plot/AverageTestEpRet.png)
    - ![LossPi](./out/test_exp_0_gauss_0p05/plot/LossPi.png)
    - ![LossQ](./out/test_exp_0_gauss_0p05/plot/LossQ.png)
    - B=2, M=0, G=1
- 1000, 0.1
    - ![AverageEpRet](./out/test_exp_1000_gauss_0p1/plot/AverageEpRet.png)
    - ![AverageTestEpRet](./out/test_exp_1000_gauss_0p1/plot/AverageTestEpRet.png)
    - ![LossPi](./out/test_exp_1000_gauss_0p1/plot/LossPi.png)
    - ![LossQ](./out/test_exp_1000_gauss_0p1/plot/LossQ.png)
    - B=1, M=1, G=1
    - I forgot that I cut s20, e1000, v0.1 short at 13 epochs. That was the run that started good from the very first epoch.
- 1000, 0.05
    - ![AverageEpRet](./out/test_exp_1000_gauss_0p05/plot/AverageEpRet.png)
    - ![AverageTestEpRet](./out/test_exp_1000_gauss_0p05/plot/AverageTestEpRet.png)
    - ![LossPi](./out/test_exp_1000_gauss_0p05/plot/LossPi.png)
    - ![LossQ](./out/test_exp_1000_gauss_0p05/plot/LossQ.png)
    - B=1, M=2, G=0
- There is a consistent correlation between LossPi decreasing into the negative, and good performance. But LossPi does not fluctuate up and down in correlation with the Mixed performance; little bumps and plateaus if anything.
- The relationship between LossQ and performance is less clear: LossQ is almost always increasing in the positive, and in _most_ cases, a higher rate of increase correlates with good performance
    - Apparent counterexample: 1000, 0.1, blue curve
    - This just baffles me even more...why would performance _improve_ when an error minimisation objective _increases_!? I can only hand-wavily explain it by the complexities of actor-critic dynamics.
- This is all very uncertain but I would lean towards exploration steps helping
- The noise std is unclear: when you identify by seed, it actually turned one Mixed to Bad and one Bad to Good for 0 exploration, and one Good to Mixed for 1000 exploration
- The big positive to take away here is that we can get good performance sometimes!
    - To totally clear my doubt though, I need to see that cheetah running with my own eyes...
- I can't say I'm surprised at the extreme variance in performance, my burning question now is: how does SU achieve such good consistency?

Running seed 20, e1000, v0.1 to complete the result

- Interesting, it isn't replicating the run I have down as the same!
- It's possible running through vscode debugger is compromising the random state repeatability...but this hasn't been a problem before

Next

- Check for random seed consistency
- Add experiment naming to logdir procedure and logger object
- Add exploration params to `ddpg` arguments
- Set up argument parsing in `main`

## 2019.06.15

Adding exploration aprams to `ddpg` arguments

- `noise_scale=1.0`
    - Standard normal distribution
- `exploration_steps=None`
    - Either `0` or `None` seems like an appropriate default. I don't want to set a non-zero integer because I think it should depend on the value of `steps_per_epoch`.

Generalising stochastic process API and integrate the current independent normal sampling

Setting up argument parsing in `main`

Modifying log directory creation to use experiment names and sub-directories

Checking random seed consistency

- Ok, I'm going to run seed 20 e1000 v0.1 twice for one epoch
- It's plausible that the changes we just made will make it different to any previous runs, so we may have to test at least two fresh runs for consistency
- Result
    - The first run stats are identical to `2019-06-12-15-58-06` (except run time)
    - The second run stats are different
    - So my most likely explanation is that the random state is not perfectly reset between calls of `ddpg` with a persistent instance of the interpretor (i.e. the loop over seeds in `main.py`)
        - Supported be re-running it and getting the same _corresponding_ results (i.e. first run matches first run and second run matches second run)
        - It is likely this could be fixed by changing or adding some call for TensorFlow or one of the other interfaces using random state. I already have `tf.reset_default_graph()` at the top of `ddpg()` and `env.close; sess.close` at the bottom
        - A workaround is to use a bash script to iterate random seeds

Trying out holding actions for 10 environment steps. Also running 10 test episodes.

- Reasoning about the HalfCheetah environment, I see how it could make more sense to have the smallest interact resolution but shorten the episode length, because the desired behaviour is very repetitive. Let's try that next and compare to this.
- Hmm, Pi and Q loss are very high (10s) so I doubt this works well, but it could be quite sensitive to the length of hold. This is only the first seed.

Trying out episode length shortened to 150 steps

- Aside: commenting out the parallel threads config
    - Seed 0 is certainly different but not widly different. I think the difference may increase over time as randomness compounds. Time is 10-20 seconds shorter per epoch for this run but I can't be sure that's not coincidence.
    - Uncommenting again, and changing threads from 1 to 4, no significant change in time.

## 2019.06.16

Running 10 seeds (0 to 90) with episode length of 150

## 2019.06.17

Reviewing shorter episode length results

- 6 Bad, 1 Borderline, 3 Mixed
- Repeats the correlations observed in the previous experiments between decreasing pi loss, increased rate of increase in q loss, and good performance (relatively). The Borderline case has increasing pi loss, but at a slower rate than the Bad cases. 
- Overall I'm not getting anything new from this experiment. Shortening the episode length seems to have little consequence in terms of the goal of consistently good performance.

## 2019.06.19

Reading DDPG paper in detail

- Same hyperparameters across tasks
    - Suggests that if we have the right implementation, it shouldn't be catastrophically sensitive to hyperparameters (esp. if we match the author settings)
- Cartpole swing-up: should try
- "Because DDPG is an off-policy algorithm, the replay buffer can be large, allowing
the algorithm to benefit from learning across a set of uncorrelated transitions."
    - Yeah, I think because each transition is evaluated by the critic anew, the age of samples is not a problem. The buffer size is a relatively minor concern; keeping it at 1 million should be OK.
- Are we copying the target variables correctly in the initial? Variable scopes could make all the difference.
- "In the low-dimensional case, we used batch normalization on the state input and all layers of the μ network and all layers of the Q network prior to the action input (details of the networks are given in the supplementary material)."
    - Check this
- "...we used an Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia"
    - So it doesn't always make the most sense
- Action repeats were only used for the high-dimensional (pixels) case
- "results are averaged over 5 replicas"
    - Seeds?

## 2019.06.20

Reading DDPG paper in detail

- Based on Figure 2, batch normalisation gives overall improvement, but not a lot. For Cheetah, there is no significant difference
- Pendulum swing-up is easier -- let's try to get hold of that environment in the continuous domain
- In some problems (especially simple domains), Q values tend to track return well, but not Cheetah.
- Should try leaving action input to the second hidden layer of Q

Main point is that we should go back to basics. The more elaborate things we have been trying may improve average performance (_may_), they may make it more robust, but we should still see good enough signs that the implementation is correct without all the frills.

Running `Pendulum-v0` on seed 0

- Why not, I'm curious. But let's not get into the weeds yet.

Next

- Review notes on paper review
- Check and think about action limits placed on actor output
- Restore OU noise process

## 2019.06.22

Reviewing notes on paper review

- You know, the Ornstein-Uhlenbeck process `dt` value is going to affect it a lot.
    - Take this example from (here)[https://www.pik-potsdam.de/members/franke/lecture-sose-2016/introduction-to-python.pdf]; implementation is mathematically the same as mine: 

        ```
        t = np.linspace(t_0,t_end,length)  # define time axis
        dt = np.mean(np.diff(t))
        ```

    - So we have some duration and divide it by the number of steps...how much time is one step in the simulator?
    - Ok if you go `env = gym.make('SomeEnv-v0')` and then `env.env.dt` you get it. For `HalfCheetah-v2` and `Pendulum-v0` at least, `dt = 0.05`.
    - Setting default `dt=.05`
- It occurs to me that we should log both critic and Q-target outputs as another check of target updates working correctly

Action limits placed on actor output

- Output activation is tanh, meaning each element is between -1 and 1
- We multiply this by `action_space.high` which stores the upper limit of actions in each dimension
- `action_space.low` is the negative of `action_space.high` as far as we have seen
- For example `Pendulum-v0` has one action dimension with `action_space.high = [2.]`. So output will be scaled to [-2., 2.]. This makes sense.
- The general way to do it would be `(h - l) * (o + 1)/2 + l` where h is high, l is low, o is tanh output
- But I would expect the relative magnitude of exploration noise to be the same across environments, given that they fix the noise parameters in the paper. This suggests it is better (but perhaps not essential) to scale actions after noise is added.

Testing pendulum env with different noise scales

- 20k steps
- Control: `out/noise-scale/2019-06-22-11-19-08_ddpg_pendulum-v0_s42`
- Running with noise scaled by `action_space.high`: `out/noise-scale/2019-06-22-11-27-13_ddpg_pendulum-v0_s42`
    - Started rendering for this one
    - Holy crap it's working!
    - And then it's not...
    - It went from balancing the pendulum upright every time in epoch 5, to going totally out of control in epoch 6, round and round and round with high torque
    - Sometimes it is good at keeping it still, so it does well when the pendulum is initialised near-upright, but it can't swing up
    - It doesn't seem to be learning (or it forgot) the back-and-forth technique. My intuition is that stronger exploration noise would help.
    - It's worth keeping in mind that in the paper, it looks like it took 50-100k steps to get "good". We are only running 20k.
        - Performance still increases fairly steadily for them though. It is probably a matter of "good" vs. "crushing it".
- The results are similar between the above two - both peak in performance at about 5000 interacts
- Oh...we should probably clip actions!
    - `out/noise-scale/2019-06-22-11-54-56_ddpg_pendulum-v0_s42`
    - Still peaks at epoch 5 and then gets unstable
    - No significant change
- Trying sigma = 0.3 with clipping
    - No significant change
- Trying sigma = 0.1 with clipping
    - No significant change

Next

- Adding Q value logs

## 2019.06.24

Adding Q value logs

- Ok, what is the current TensorBoard situation?
    - Line 95: `tf.summary.scalar('q_loss', q_loss)`
    - So I think we just repeat this line for different vars
- Hmm, actually, how is this going to work? We have a batch of random experiences, each with a Q value.
- Logging Q values with the epoch logger makes some sense, but TensorBoard makes less sense.
    - One reason with TensorBoard is as another verification for target values tracking critic values
- What exactly are the values in Figure 3 of the paper?
    - "estimated Q values versus observed returns"
    - Are the Q values after all training, or throughout training?
    - We have a bunch of episodes. For each episode, there will be a return; for each time step, there will be a Q value.

Testing with Q value logging (seed 42)

- Closely matches pi loss, as expected. Not an exact match because the reported pi loss value is an average of averages over each batch, and the reported q value is an average of all values in every batch
- Q loss peaks at 185 at about epoch 25 (1000 steps per epoch), then decreases substantially (to about 50) before fluctuating over shorter timescales. My most plausible explanation for the decrease is that in addition to the usual SGD pressures, the behaviour of the actor becomes more predictably bad.
- I feel like it is key, the fact that it can nail a good policy within a few thousand interactions, then crash.
    - Poor feedback (i.e. problem with Q)?
        - If so, how would it reach a good policy in the first place?
    - Poor response to feedback (i.e. problem with actor)?
        - The actor is clearly _trying_ towards the goal, it's just bad at it
        - For example, when the pendulum initialises right near upright, the actor first pushes towards upright, and when it overshoots, it pushes back
    - Breakdown of the actor-critic relationship?
        - This is hand-wavy, but what I'm getting at is that both the actor and critic, the interaction between them, results in unstable behaviour.

Critic vs. target in TensorBoard

- The quantities are clearly different
- Target has less variance _in the first 3 or so thousand interacts_. After that, there doesn't seem to be much of a smoothing effect, and the delay is not very apparent, but nor are the values identical.
- Both quantities steadily increase in variance over time, which can be explained well by the accumulation of experience, but I'm not sure if that's the sole explanation.

## 2019.06.27

Possibly important thought

- Policy gradient
    - Gradient of Q with respect to action
    - Action is actor output _plus_ noise
    - Noise is not in computation graph
    - Clearly, the action at the current time step has noise. But should the policy gradient include or exclude the noise? Does it matter (i.e. does it get differentiated to zero)?
    - Look _carefully_: paper says

        $$ \nabla_{\theta^\mu} J \approx \frac{1}{N} \sum\limits_{i} \nabla_{a} Q(s, a | \theta^Q)|_{s=s_i, a=\mu(s_i)} \nabla_{\theta^{\mu}} \mu(s | \theta^{\mu})|_{s_i} $$

        - Let's assume in good faith that there are no typos in this equation
        - Here (_here_, for this calculation) the action input to Q is $\mu(s_i)$. No noise.
        - What's the point of noise then? Well, $a_i$ (the batch action that was originally $\mu(s_t | \theta^\mu) + \mathcal{N}_t$ at some time step) is used for Q loss, _not_ pi loss. In fact, it is only used as input to Q for Q loss. The Bellman backup uses the (again, _noiseless_) target actor output.
    - Hmm, no, I still think our graph is correct.

Removing weight decay in actor

- Bad result (but very limited test)

Removing weight decay in critic

- Wow, much better
- Much more consistent on a good policy
- Still fluctuates up and down in performance a little
- The policy (at least observed in the first 37 epochs of seed 42) uses rapidly alternating, usually high force to keep the pendulum up. This doesn't seem absolutely optimal, or at least is not qualitatively optimal.
- If this is a consistent result (across seeds), it is quite remarkable...two possibilities:
    - There is something critically different about the implementation of `AdamWOptimiser`
        - IIRC widespread implementations of Adam with weight decay had a bug/feature - could success be reliant on that?
    - Weight decay parameter is sub-optimal
        - It seems unlikely that the effect would be this drastic - we have never seen performance close to this good or consistent. Then again, basic DRL like this is known to be fragile.

Given we may have hit upon the winner, what was my thinking process to get here?

- I turned on environment rendering for training as well as testing, because I wanted to see the effect of exploration noise more clearly
- Seeing the policy go from good to wildly bad (pendulum swinging around and around at high speed) made me think of gradient explosion and extreme weight values.
- If so, I figured weight decay on the actor might help. So I tried that.
- That didn't work, so I thought "why not" and instead _removed_ weight decay on the critic. I don't think this was reasoned causally, just an explorative, curiosity-driven decision.

Next

- Run pendulum on 5 other seeds with current configuration (namely no weight decay on critic)

## 2019.06.28

Running pendulum on 5 other seeds

- All (relatively) consistent, good performance!
- One of them had a bad spike towards the end.

Restoring weight decay, but reduced from 1e-2 to 1e-3. 5 seeds.

- According to the documentation for `AdamWOptimizer`,

    > It computes the update step of train.AdamOptimizer and additionally decays the variable. Note that this is different from adding L2 regularization on the variables to the loss: it regularizes variables with large gradients more than L2 regularization would, which was shown to yield better training loss and generalization error in the paper above.

- It regularises more than the standard. So I hypothesise that a lower weight decay parameter value will yield results closer to the paper. I can't actually compare because their scores are normalized, so what I really mean is that results will improve over the 1e-2 value. This also intuits from the fact that a value of 0 has turned out to be much better than 1e-2 (assumes smooth performance change with respect to the weight decay parameter value)
- Result
    - Better than 1e-2, worse than 0
    - Test return has higher variance and lower average (about -250 rather than -160)
    - Return consistently drops to around -1000 at around 25k interacts before returning to high
    - Pi loss rises to about +70 and stays there, whereas for 0 weight decay it rises then steadiliy drops into the negative
- So the two most plausible worlds here: the optimal weight decay for Pendulum is 0, or it is between 0 and 1e-3.
- It is entirely possible that the optimal weight decay for Pendulum is far from the jointly optimal weight decay across many different environments, as in the paper. Presumably, they tried multiple values and checked the overall performance before settling on 1e-2. But the fact that their chosen 1e-2 is distastrous for Pendulum in our implementation is strong evidence of a difference in the weight decay optimiser.
- The possibility of a relevant structural difference between `AdamWOptimizer` and traditional (Adam + weigh decay) still stands.

Weight decay 1e-4

- Performs between 1e-3 and 0 (only tested seed 0)

HalfCheetah redux

- Weight decay 1e-4
    - Bad
- Weight decay 0 (`AdamOptimizer`)
    - Bad

## 2019.06.30

Review

- We removed weight decay on the critic and got consistent high performance for pendulum, where before it was occasionally good but generally bad with high variance.
- We did the same for cheetah and saw no improvement.
    - Losses became very large. Pi loss was negative on the order of 1000, in one case 10,000
        - Indicates critic is overestimating the return.
        - Is the critic without weight decay easier to exploit by the actor?
    - Return broadly averaged to about -500. So an even worse situation than the current best of "mixed" performance that we found with uncorrelated Gaussian noise instead of the default OU process (see 2019.06.12).

Third environment

- Third environment's the charm: try a third environment, see which way the performance lands without critic weight decay.
- Swimmer
    - Standard DDPG is far and away the best algorithm in the Spinning Up benchmarks. Some evidence that it is relatively more likely to do well for us.
    - Let's check the parameters of the problem.
        - obs_dim = 8
        - act_dim = 2
        - 1000 steps per episode
        - Not too bad

Detour: cheetah with weight decay 1e-3 and uncorrelated Gaussian exploration noise (seed 0)

- 3 epochs in and it's doing very well (1800 test return)...let's see
- Finished now. This is a big improvement, gets around 1500 return through to the end. However, there are still a few big hiccups in return; I know this is not unheard of but it seems a bit too inconsistent still.
- Now trying 1e-3 with OU process exploration (still just seed 0)
    - Much worse - the usual average 500 with high variance
- Ok, so either there is a bug in our OU implementation, or the optimal weight decay differs between OU and Gaussian. But including the evidence from 2019.06.12 suggests that uncorrelated Gaussian exploration noise generally gives much better performance for cheetah (at least for our implementation).
- Let's see how Gaussian std 0.1 noise with 1e-4 weight decay compares to 1e-3
    - Worse, high variance, average around -400 but gets into positive hundreds

Running Swimmer-v2, wd=1e-3, Gaussian std=1e-1

- Inconsistent, generally doesn't look very good
- Benchmarks indicate it takes much longer to get good, say 0.5-1 million interacts. I would still expect better than this over 100,000 interacts, but with a single seed, there's not enough evidence.
- Running 100 epochs, 10,000 steps/epoch, episode length 200
    - Got interrupted, will need to run again. Return was steady around 30 though, maybe reducing the episode length is bad


## 2019.07.06

Running swimmer as last time but with normal episode length

Review

- I think we should double down on as-exact-as-possible replication of the original method. In particular that means implementing weight decay the old-school way, and feeding in actions at the second hidden layer of Q. Maybe even setting the network initialisation method the exact same.

## 2019.07.07

Implemented standard weight decay via direct L2 loss on `ac_vars`

Testing cheetah with weight decay 0.01, normal noise std 0.1

- This has everything the same as the best-to-date cheetah run, except new weight decay
    - `out/critic_weight_decay/2019-06-30-12-06-05_ddpg_halfcheetah-v2_s0`

## 2019.07.08

Moving action input to second hidden layer for Q

Customising parameter initialisation of MLPs

- `tf.layers.dense()` takes `kernel_initializer` and `bias_initalizer` as arguments
- Paper:

    > The final layer weights and biases of both the actor and critic were initialized from a uniform distribution [-3 × 10^-3, 3 × 10^-3 ]

    > The other layers were initialized from uniform distributions [-1/sqrt(f), 1/sqrt(f)] where f is the fan-in of the layer

- The `glorot_uniform_initializer` seems to be the default
- "fan-in" is the number of inputs to a hidden unit. So (obs_dim, 400, 300)

As far as I know and intend, the implementation now matches the paper.

Cheetah, standard settings, seed 0

- Note that's OU process noise, which has always been problematic. So not expecting good things.
- Epoch 1: jiggle vigorously!
- Epoch 2: drop it like it's hot (or: drop it like there's no reward)
- Epoch 3: leap before you faceplant
- Epoch 4: sit!
- Epoch 5: stand tall
- Epoch 6: leap before you faceplant redux
- Epoch 7: poised
- Cancelled

Normal noise

- Same deal. Not moving forward!

## 2019.08.04

Review

- Before coming back I was considering checking a solution, on the grounds that it offers more learning value than continuing to experiment solo at this point. However, reading through recent notes I don't think I'm quite done. For one, we ought to go back to Pendulum to test the new as-exact-as-knowable implementation on the simplest environment, rather than toil away at the more complex Cheetah.
- The "greatest hits" of our journey have been
    - Using uncorrelated Gaussian exploration noise rather than OU process. However, I expect this is due to a bug in the OU process implementation, or other parameters are not set right with respect to OU, or some other bug is interfering.
    - Reducing weight decay. However, it is no longer clear that this is the way to go now that I have changed the weight decay to the "classic" formulation rather than `AdamWOptimiser`. This is a key thing to investigate, starting with Pendulum.
- I am concerned about the recent change to as-exact-as-possible introducing bugs. We didn't check it that thoroughly, initially. It's hard work to get back into but I think it's worth another once-over and cross-check against the paper details, check of how TensorFlow functions work, etc.

## 2019.08.05

Reading through all code

- Hmm, so I know we flip-flopped on `pi_loss` depending on `q_pi`, then `q`, then back to `q_pi`. I think `q_pi` is the right _value_ to use, but my reaction is that backpropagating this loss would result in the `q` parameters being changed as well as the `pi` parameters. We just want the `pi` parameters to change here. Watch this space.
    - In `minimize` function there is a `var_list` argument:

        > Optional list or tuple of Variable objects to update to minimize loss. Defaults to the list of variables collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES

    - We need to ablate this bad boy
- `q_vars = [v for v in ac_vars if 'q' in v.name]`
    - Is this going to include `q_targ` incorrectly?
    - No, because `ac_vars` excludes target vars
- Not sure of clipping the actions to their limits after noise is added is part of the original algorithm - isn't this a change introduced by TD3?
    - I think it is, but TD3 also uses clipped noise. So we can take it out to be purist, but otherwise it should generally be good.

Checking new paper-detail changes

- Fan-in is the number of inputs to one hidden unit. For an MLP, this is the length of the input vector. `out.shape[1]` seems correct since `out` is the previous output, in turn the current input, and the zeroth dimension is batch size, so the first dimension is the length.

Running Pendulum with new paper-detail changes (seed 0)

- Gaussian exploration noise, std 0.1
- Just to emphasise: standard weight decay with parameter 0.01
- Looks fine! Mildly varying but generally good swing-up, generally -200 to -100 test return
- As good a time as any to note: user leaderboard for best 100-episode performance (presumably test average) on Pendulum-v0 ranges from -152.24 +- 10.87 to -123.11 +- 6.86. We aren't getting that exactly-upright-every-time performance, but close to it.
    - Epoch 35 to 85: -178.26 +- 94.75.
    - Epoch 50 to 100: -170.91 +- 95.25
    - Epoch 75 to 100: -151.42 +- 31.15
- There is the occasional drop, e.g. epoch 55 (-511) and 64 (-661)
- Now, we haven't made any substantive changes to the code since the last cheetah run, so we should still consider that problematic. So now I will implement the backprop change and test Pendulum again.

Running Pendulum with `actor_minimize` `var_list=pi_vars` (seed 0)

- Looks better - tighter angles
    - Update: tighter angles only sometimes
- Comparative results
    - Epoch 35 to 85: -160.01 +- 40.31
    - Epoch 50 to 100: -154.50 +- 39.29
    - Epoch 75 to 100: -151.86 +- 30.59
- So towards the end, about the same, but better earlier on, without major hitches. This shouldn't be given much weight because it's only one seed, but it almost surely is not worse.

Running Cheetah with `actor_minimize` `var_list=pi_vars` (seed 0)

- It just isn't _moving_. It stops after a few hundred steps. The best it does is a handstand.
- Cranking the exploration noise to 0.5 std
    - Yup, exploration noise must control the "do stuff" parameter to an extent, but 0.5 may be overkill - it is doing much better but is unreliable and erratic (4 epochs in)
    - Case in point, it is currently succeeding by sliding along its back.
    - Fairly stable; one collapse in return (epoch 18) where it just flipped over and stopped.
- Goldilocks: 0.2 std
    - Similar to 0.1
- 0.3 std
    - Looked ok, but not beyond 1000 (didn't see visual behaviour)
    - Seed 10 and 20
- 0.2 std
    - Seed 10 and 20

## 2019.08.06

Cheetah seed 0,10,20 std 1.0

- So it's doing ok again, but, again, it likes to move upside down
    - Who am I to judge its preferred locomotive orientation?

## 2019.08.10

Pendulum std 1.0, seed 10

- Comparing this given it gave better results than 0.1 for cheetah
- Seed 10 for a change
- Doing fine
- I think if there's a common thread for my algorithm between cheetah and pendulum, it's laziness. The policy for pendulum rarely strives to be perfectly upright, even though that seems possible to learn in principle. Similarly for cheetah, it might learn to move on its back and it never breaks into the higher reward space of upright sprinting. It's like a mild optimiser; it settles for "close enough". It would be interesting to get to the bottom of this when comparing to a canonical solution.
    - However I may be speaking on too little data. This run is converging on upright as I write. But even if so, I'm not confident it will be a stable policy for long.
- It got closer than I've ever seen for longer than I've ever seen to perfectly upright, but still diverged from this after a time.
- In later epochs it did that technique of rapidly switching the direction of torque, which I haven't seen in a while.
- Comparative results
    - Epoch 35 to 85: -150.15 +- 35.81
    - Epoch 50 to 100: -158.92 +- 34.38
    - Epoch 75 to 100: -168.14 +- 29.86
    - Got worse! Started better than the previous two I recorded the same stats for. Then went to second place. Variance is consistently lowest though.

Decision to check the solution

- I've decided at this point, it's more valuable to check the solution than perfect the implementation in the dark. I have learnt much, and the implementation seems decent. Satisfied for a first try. Got pendulum working well. Cheetah is a mixed bag.
- So here goes...

Spinning up DDPG implementation

- MLP function OK
- Ah, `get_vars` abstracts variable retrieval based on name matching
- `mlp_actor_critic`
    - Takes `act_dim` from `a` rather than `action_space` (should be fine)
    - `act_limit` is `action_space.high[0]` where I use `action_space.high` (their assumption is that the bound is the same for all dimensions - not sure why)
    - For q, action fed at first layer with state (knew this, different to paper)
    - Action concatenated to state on last (-1) axis (I use axis 1, should be fine in this case)
- `ReplayBuffer`
    - A class implementation, makes sense for clean code
- Default parameters that differ
    - `polyak=0.995`
        - Hopefully it swaps the variables relative to this...yes, confirmed
        - Equivalent to 0.005 vs. our 0.001
    - `pi_lr=1e-3`
        - 1e-4 for us (and paper)
    - `batch_size=100`
        - 64 for us
- No `reset_default_graph()`
- `EpochLogger` first (ours has `gym.make` before that)
- Does not use `random` library (just `np.random`)
- `env.seed` not set
- TF seed before NP seed
- `actor-critic` block is the same
- `target` block is the same
- `backup` is same except for "done" stopper (knew this, different to paper AFAIK)
- `pi_loss` same
- `q_loss` same (without weight decay) except subtraction is swapped, which shouldn't matter
- `target_update` is set up quite differently but it seems to be effectively the same
    - I don't know, there could be tensorflow voodoo at work
    - Makes use of `tf.group` to bundle all the target updates, where I just use the Python list
    - Uses `tf.assign` as I do (just object vs. function syntax)
- I am happy that I created `targ_init` and `targ_update` very similar to this implementation even though I had mostly forgotton that part of it and had to reason about it carefully on my own.
- Specifying `var_list` for the pi optimiser is a match, but they also specify it for q. Pretty sure this doesn't matter, but worth checking.

## 2019.08.11

Continuing review of Spinning Up DDPG implementation

- Big one: no weight decay!
- Order of ops before training
    - This: placeholders, actor-critic, target, replay buffer, backup, loss, optimisers, target update, target init, session
    - Mine: placeholders, actor-critic, target, target init, backup, target update, optimisers, buffer, session
- Action retrieval function
    - They include the execution of pi in the separate function
        - No `squeeze`
    - `a + std * normal(1.0)` where I use `a + normal(std)`
        - Always wondered about the equivalence of these...
        - Yes, equivalent because of standard normal distribution conversion (z-score) http://mathworld.wolfram.com/NormalDistribution.html
    - Noise is not scaled by action space limit
    - Default noise scale is 0.1...the plot thickens
- Exploration steps is a thing! Default 10,000
- Ah, now this is interesting:

    ```python
    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    d = False if ep_len==max_ep_len else d
    ```

    - Makes sense
- Ah, interesting also: bundle gradient updates to the end of each trajectory/episode (this is informed by TD3)
- Test procedure OK - nice use of `get_action` function with 0 noise to avoid repetition
- Buffer OK
- Pi is updated regardless of exploration steps

Thoughts

- Well, my takeaway is that the differences are pretty minor, and nothing jumps out as being the reason for my mixed success. Happy with how close I got.
- I can't find any implementation online from the original authors

Testing most recent pendulum run with specification of `q_vars` for `critic_minimize`

- Identical over first 10 epochs (stopped)

Removing action-limit-scaling on noise and changing std to 0.1

- Epoch 35 to 85: -156.65 +- 36.29
- Epoch 50 to 100: -163.79 +- 34.45
- Epoch 75 to 100: -173.69 +- 30.37
- Similar to previous, but a bit worse
- No outliers

Cheetah std 0.1 with above changes, seed 10

- Did well! Performance ended very high (~2000), no big hiccups after getting good (6 epochs)
- My takeaways from the very limited data I have
    - Noise scale affects stability and the risk of "lock in" to a very suboptimal but OK solution
    - When I wasn't getting promising results with lower scale (0.1), I thought a higher scale would allow it to explore more, increasing the chance of hitting the right actions. I think this is true to an extent, but large noise can get it stuck in an OK but not great solution early on (case in point: cheetah running on its back).
    - Lower scale means slower learning, but more stable and incremental improvement.
- Seed 0 and 20
    - Gee, neither take off! Hang around 0 return. Maybe they just need more time? 100k interacts is not much in this context.

