Neural Turing Machine
======
Pytorch Impelementation of [Neural Turing Machine](NTM), the model part is
referenced from [loudinthecloud/pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm)

The related post is [here](https://sunprinces.github.io/learning/2018/05/beyond-rnn---lec1/) (belong to the series of **Beyond RNN**)

## Additional Feature

* Add tensorboard to track training curve and the output posterior of validation sample in real-time
* **Copy** task detailed analysis

## Copy Task

### Training

The red line is evaluated on the sequence with length 30, and the blue line is
training sequence with random length 3 ~ 20

![cost](/fig/0320-cost.png)

![loss](/fig/0320-loss-curve.png)

Basically, the model learned this task after seeing 25k training examples

![prediction in the middle round](/fig/0320-prediction-middle.png)

In the middle round, NTM haven't learned yet. The blurry posterior indicated the
unsureness.

![prediction in the final round](/fig/0320-prediction-final.png)

Eureka!

### Analysis

![R/W weight](/fig/head-weight.png)

The parallel line indicated the w/r position, and there is a clear switch in the
middle.

![Memory usage](/fig/memory-usage.png)

Trained on 3 ~ 20-length sequence, predict on 50 ~ 220-length sequence

![Error Curve](/fig/error-curve.png)

According to the error curve, the bottleneck length is the number of memory
cell, and also show the ability of generalization to longer sequence of NTM.

## TODO
- [ ] Reproduce repeat-copy task
- [ ] Think some interesting application using N-gram
