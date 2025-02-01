# I. Foundations and Core Techniques in Deep Learning

## Backpropagation and Early Neural Networks

- **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).**  
  “Learning Representations by Back‐propagating Errors.”

- **Werbos, P. J. (1981).**  
  “Applications of Advances in Nonlinear Sensitivity Analysis.”  
  *(An early discussion of ideas that eventually became backpropagation.)*

- **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).**  
  “Gradient-Based Learning Applied to Document Recognition.”  
  *(Classic paper describing LeNet and early convolutional ideas.)*

- **Glorot, X., & Bengio, Y. (2010).**  
  “Understanding the Difficulty of Training Deep Feedforward Neural Networks.”  
  *(Introduces improved initialization methods.)*

## Optimization & Regularization Techniques

- **Kingma, D. P., & Ba, J. (2014).**  
  “Adam: A Method for Stochastic Optimization.”

- **Ioffe, S., & Szegedy, C. (2015).**  
  “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.”

- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).**  
  “Dropout: A Simple Way to Prevent Neural Networks from Overfitting.”

# II. Convolutional Neural Networks and Computer Vision

## Key CNN Architectures

- **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).**  
  “Gradient-Based Learning Applied to Document Recognition.”  
  *(LeNet for handwritten digit recognition.)*

- **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).**  
  “ImageNet Classification with Deep Convolutional Neural Networks.”  
  *(AlexNet.)*

- **Simonyan, K., & Zisserman, A. (2014).**  
  “Very Deep Convolutional Networks for Large-Scale Image Recognition.”  
  *(VGG.)*

- **Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015).**  
  “Going Deeper with Convolutions.”  
  *(Inception.)*

- **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**  
  “Deep Residual Learning for Image Recognition.”  
  *(ResNet.)*

# III. Sequence Models, Recurrent Networks, and Attention

## Recurrent Architectures & Sequence-to-Sequence Models

- **Hochreiter, S., & Schmidhuber, J. (1997).**  
  “Long Short-Term Memory.”

- **Cho, K., Van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).**  
  “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.”

- **Sutskever, I., Vinyals, O., & Le, Q. V. (2014).**  
  “Sequence to Sequence Learning with Neural Networks.”

- **Bahdanau, D., Cho, K., & Bengio, Y. (2015).**  
  “Neural Machine Translation by Jointly Learning to Align and Translate.”

## Attention and Transformer Models

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).**  
  “Attention Is All You Need.”  
  *(Introduces the Transformer architecture.)*

# IV. Generative Models and Unsupervised Learning

## Generative Adversarial Networks and Variants

- **Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).**  
  “Generative Adversarial Nets.” *(GANs.)*

- **Radford, A., Metz, L., & Chintala, S. (2015).**  
  “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.”  
  *(DCGAN.)*

## Variational and Latent-Variable Models

- **Kingma, D. P., & Welling, M. (2013).**  
  “Auto-Encoding Variational Bayes.”  
  *(VAE.)*

- **Rezende, D. J., Mohamed, S., & Wierstra, D. (2014).**  
  “Stochastic Backpropagation and Approximate Inference in Deep Generative Models.”

# V. Reinforcement Learning  
*(Key papers as highlighted by Spinning Up and related influential works)*

## A. Value-Based Methods

- **Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015).**  
  “Human-Level Control through Deep Reinforcement Learning.”  
  *(DQN.)*

- **Van Hasselt, H., Guez, A., & Silver, D. (2016).**  
  “Deep Reinforcement Learning with Double Q-Learning.”  
  *(Double DQN.)*

- **Bellemare, M. G., Dabney, W., & Munos, R. (2017).**  
  “A Distributional Perspective on Reinforcement Learning.”  
  *(Distributional RL.)*

## B. Policy Gradient and Actor–Critic Methods

- **Sutton, R. S., McAllester, D., Singh, S. P., & Mansour, Y. (1999).**  
  “Policy Gradient Methods for Reinforcement Learning with Function Approximation.”

- **Schulman, J., Levine, S., Moritz, P., Jordan, M., & Abbeel, P. (2015).**  
  “Trust Region Policy Optimization.”  
  *(TRPO.)*

- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).**  
  “Proximal Policy Optimization Algorithms.”  
  *(PPO.)*

- **Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016).**  
  “Asynchronous Methods for Deep Reinforcement Learning.”  
  *(A3C.)*

- **Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2015).**  
  “Continuous Control with Deep Reinforcement Learning.”  
  *(DDPG.)*

- **Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014).**  
  “Deterministic Policy Gradient Algorithms.”

- **Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).**  
  “Soft Actor–Critic: Off–Policy Maximum Entropy Deep Reinforcement Learning.”  
  *(SAC.)*

## C. Integrated and Advanced RL Methods

- **Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Silver, D., & Sutton, R. S. (2017).**  
  “Rainbow: Combining Improvements in Deep Reinforcement Learning.”  
  *(Rainbow DQN.)*

- **Fujimoto, S., Hoof, H., & Meger, D. (2018).**  
  “Addressing Function Approximation Error in Actor–Critic Methods.”  
  *(TD3.)*

# VI. Additional and Emerging Topics

## Meta-Learning and Few-Shot Learning

- **Finn, C., Abbeel, P., & Levine, S. (2017).**  
  “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.”  
  *(MAML.)*

## Self-Play and Mastery in Games

- **Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T., Leach, M., Kavukcuoglu, K., Graepel, T., & Hassabis, D. (2016).**  
  “Mastering the Game of Go with Deep Neural Networks and Tree Search.”  
  *(AlphaGo.)*

- **Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Steiner, A., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., & Hassabis, D. (2017).**  
  “Mastering the Game of Go without Human Knowledge.”  
  *(AlphaGo Zero.)*

## Memory-Augmented Neural Networks

- **Graves, A., Wayne, G., & Danihelka, I. (2014).**  
  “Neural Turing Machines.”

- **Weston, J., Chopra, S., & Bordes, A. (2015).**  
  “Memory Networks.”

