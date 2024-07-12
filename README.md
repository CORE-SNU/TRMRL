Task-relevant Meta-reinforcemen Learning (TRMRL)
====================================================
![TRMRL](https://github.com/user-attachments/assets/90d86e2b-6cbf-4d6d-92fe-322480da2a58)
This repository includes an official PyTorch implementation of [**On Task-Relevant Loss Functions in Meta-reinforcement Learning**][paperlink], which will appear in [L4DC 2024][l4dclink]. Our implementation is based on [PEARL][pearllink] implementation, and use a code from [dm_control][dmcontrollink] to generate a task and visualize the learned results.
## 1. Requirements

We have tested that the implementation successfully runs under the following dependencies:

- **Python**

- **[Gym][gymlink]** 0.26.2

- **[Pytorch][pytorchlink]** 1.12.1

- **[dm_control][dmcontrollink]** 1.0.9


## 2. Training
To train a TRMRL agent on a bipedal walker task, run
```
$ python trmrl.py ./config/walker.json --gpu=[your-gpu-id]
```


## 3. Test
To test the agent learned trained on the bipedal walker task, you may simply run
```
$ python sim_policy.py ./config/walker.json ./output/walker/[path-to-log-directory] --video --num_trajs=2
```


[pearllink]: https://github.com/katerakelly/oyster
[dmcontrollink]:https://github.com/google-deepmind/dm_control
[gymlink]: https://github.com/openai/gym/
[pytorchlink]: https://pytorch.org/
[paperlink]: https://proceedings.mlr.press/v242/shin24a/shin24a.pdf
[l4dclink]: https://l4dc.web.ox.ac.uk/home

