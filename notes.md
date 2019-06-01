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
