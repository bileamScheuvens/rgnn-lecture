# Exercise 1

## LSTM
### Epoch 10:
Initialization: ‘but the night will be too sho...

Tolkien:
 --- ‘but the night will be too short,’ said gandalf. ‘i have come back here, for i must have a little peace, alone. you should sleep,

LSTM model:
 --- ‘but the night will be too shot te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te te

### Epoch 100:

Initialization: at last they came out of shado...
Tolkien:
 --- at last they came out of shadow to the seventh gate, and the warm sun that shone down beyond the river, as frodo walked in the gla

LSTM model:
 --- at last they came out of shador and the store and the store of the store and the store of the store and the store of the store and

## TCN
### Epoch 10:
Initialization: ‘then why did you not say so a...

Tolkien:
 --- ‘then why did you not say so at once?’ said bergil, and suddenly a look of dismay came over his face. ‘do not tell me that he has

TCN model:
 --- ‘then why did you not say so ayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy

### Epoch 100:

Initialization: ‘man!’ cried pippin, now thoro...
Tolkien:
 --- ‘man!’ cried pippin, now thoroughly roused. ‘man! indeed not! i am a hobbit and no more valiant than i am a man, save perhaps now

TCN model:
 --- ‘man!’ cried pippin, now thoro!!!

(((??


uu)))!!!


gggrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

## Transformer
### Epoch 10:
Initialization: ‘you are peregrin the halfling...

Tolkien:
 --- ‘you are peregrin the halfling?’ he said. ‘i am told that you have been sworn to the service of the lord and of the city. welcome!

Transformer model:
 --- ‘you are peregrin the halfling the the the the the the the the the the the the the the the the the the the the the t the the t the

 Initialization: pippin lifted it and presented...

### Epoch 100:
Tolkien:
 --- pippin lifted it and presented the hilt to him. ‘whence came this?’ said denethor. ‘many, many years lie on it. surely this is a b

Transformer model:
 --- pippin lifted it and presented the hilt he him. ‘came the seany,’ said inet lan ore our of the ling oredred in pan in the win the l


# Exercise 2

## b
Images of a VAE with higher beta seem to have more contrast and look sharper but a little less coherent/realistic, while the VAE with beta 0 produces blurry images of plausible looking faces.
Due to computational constraints it is likely that neither model was trained to convergence, and it is thus unclear but likely that this effect would persist or even amplify with more training.
The beta value enforces more distinct structure in the latent space, while a vanilla VAE is free to learn an average of all possible faces.

## c
For the vanilla VAE varying individual dimensions produces drastically different faces, but it is rarely clear how to interpret this change as opposite faces have little in common.
Additionally many dimensions are similar in their output.
For the VAE with beta 20, the distinction is stronger and for many dimensions one can make an educated guess about what the dimension represents, sunglasses vs hats (dim 0 and 3), hair vs bald (dim 5) or skin color (dim 18).

## d
Multiple dimensions contain some information about rotation/ facial orientation, but for our model, dimension 11 of the beta20 VAE was the most prominent.
Still, negative pertubations have little effect (possibly due to the sigmoid activation), leaving the face mostly centered. Positive pertubations however rotate the face as desired. 