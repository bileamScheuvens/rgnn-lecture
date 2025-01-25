1 b)
Both environments have the markov property. In the simple gym this is obvious, in the lunar lander gym it stems from the fact that the state includes not just position, but velocity. Thus no memory is required and an RNN is not necessary.

2 a)
As expected, the open scenario is by far the most reliable, with MSE of around 1.72 and stable predictions in the video with strong segmentation of objects and background.
In both the auto and black case, the predictions are more blurry, with significant attention weight in most slots going to the area around the objects. 
Visually the auto case looks slightly better, but the MSE is marginally higher than the black case with 26.81 vs 23.11.

2 b)
In all cases the performance is slightly worse than with more teacher forcing, with the black scenario being least affected by this change.
Interestingly, the tracking seems more centered on the object, rather than anticipating movement in many cases.

2 c)
With 8 slots, the slots start competing for the objects and we observe flickering and unstable tracking, where from one frame to another the slots switch between different objects.
This behavior is more pronounced whenever (partial) occlusion occurs. The MSE is also significantly higher with even the open scenario scoring 16.5.

2 d)
Without the inner loop performancee falls dramatically, with MSE 40.83 for the open and >80 for the other scenarios.
We observe much flickering and highly blurry attention maps. The model fails to capture the behavior of the objects.