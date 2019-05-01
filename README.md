# Reproducing DDPG in Python with TensorFlow

A personal project to reproduce the deep deterministic policy gradient (DDPG) algorithm for continuous control under reinforcement learning (RL) [1]. It is considered reproduced if it obtains similar results to [1], or else achieves respectable average return in multiple popular continuous control toy environments.

The implementation is in Python using the TensorFlow library. It uses the [OpenAI Gym](http://gym.openai.com/) API to handle environments, and is heavily based on the author's memory of the [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) library. However, as a rule no DDPG or related algorithm implementations are directly read while the project is in progress.

[1] Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).
