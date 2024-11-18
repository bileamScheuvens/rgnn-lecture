## 1
### d

## 2

### c
The model performs well for all static or consistently changing environments, such as the trees, the path or the player when standing still.
Since inputs are not available to the model, it cannot predict the player's movement leaving it to be very unsure of player position and completely dropping the sprite after just a few frames.

### d
Without attention the model has no concept of spatial relationships between patches, it merely learns that the player is at the center of the image and outputs this as a static prediction. Attention helps put individual activations into context, which is necessary to understand surrounding.

### e
Without positional embeddings the model performs very poorly and converges to point of treating all patches equally. The output is a homogenous blur resembling the texture of what could be a tree.

### f
Without a thorough gridsearch we found it difficult to make definite conclusions about the architecture. Since jobs on the cluster are strictly time limited, more than limited exploration of the search space was not possible.
More complexity(n_layers/heads) seemed to be beneficial, as well a larger learning rate. Increasing the number of frames in the input sequence slowed down convergence, it is unclear if the final performance would be better since the model was not able to converge within the time limit.
The same can be said for increasing the model complexity drastically.
The patch size seemed optimal at 16x16, as 32x32 does not divide the image evenly and 8x8 or 4x4 increase memory demands and slow down convergence significantly in terms of time, while being more sample efficient.
Weight decay, mixed precision and gradient_accumulation_steps were not explored.
Batch size was fixed as the scaling laws dictate the ratio to the learning rate, which was optimized.

For the final configuration we obtain the default configuration with the following adjustments:
"num_heads": 4,
"layers": 6,
