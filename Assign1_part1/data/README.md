# Assignment 1 
## Part1: Cifar and Minst
---
## Part2: Kaggle Competetion of Image Classification

Competetion Results: 99.176% on public Dataset and 99.319% on private(final) Sataset


### Methods used
The model includes two part: 
1. the 4 usual conv layers followed by 2 fully connected layer. 
  

  * CNN feature maps: 3 - 32 - 64 - 128 - 256
  * FCNN parameters: 256 * 4 * 4 - 500 - 43
  * Kernel size: 5, 5, 3, 3
  * Leakey Relu(better performance than relu) as activation function, Batch normalization after each conv layer, dropout layer between (a) CNN block and Linear block (b) 2 linear layer
  * Inspired by the paper about improved lenet-5, instead of adding maxpooling after every CNN layer, only two maxpooling layers are implemented between 2nd layer and 3rd layer and after 4th layer. According to the paper, this design can extract the info of same feature deeper, and it does improve the performance. (My final model is more or less an invariant of the improved lenet-5 model, however, instead of use the same feature numbers for every CNN layer, I found the conventional increasing channels has better performance.)


2. Spatial transformer
  According to the webpage http://torch.ch/blog/2015/09/07/spatial_transformers.html, spatial transformer can increase the performance of the model. The spatial transformer network implemented in my model has feature maps 3 - 8 - 12 - 32, with kernel size 5 and 7

3. Other methods tried but not used
  * Local contrast nomalization. This method is once mentioned in the class but did not enhance the performance

---

### Training
1. Gradient descent

  In my final model I used Stochastic Gradient Descent for training, with $lr = 0.0025$, $momentum = 0.9$.
  * After experiments, small learning rates have better performance. The range is $[0.002, 0.005]$. Generally speaking,  0.003/0.004 has the best performance. However, the best model is trained with $lr=0.0025$
  * I have also tried Adam Gradient Descent, it has no  better performance than the usual stochastic gradient descent
  * I have also tried scheduler to perform learning rate decay. However, it was not benefeciary to my model.

2. Epochs
 Because of the low learning rate used, it generally takes 50-60 epochs to converge. In most of the cases, Models after epoch 60 is overfitted according to the training loss and validation accuracy. However, because of the dropout layer, it is not certain. (Sometimes the model reach the highest validation accuracy around 70 epochs) In practice, I set the epochs to 80 to avoid losing some good models. Epochs after 60m nevertheless, are in most cases a waste of time.

3. Ensemble

  The final model is an ensemble of 3 models during with the best validation accuracy. 

---

### Evaluation 
  My approach got the highest test accuracy of 99.176% (on Kaggle)

  Note that the training output followed is not the traning output of my best model

---
### References
  * The power of Spatial Transformer Networks.  http://torch.ch/blog/2015/09/07/spatial_transformers.html
  * A Lightweight Model for Traffic Sign Classification Based on Enhanced LeNet-5 Network.  https://doi.org/10.1155/2021/8870529
  * Pytorch tutorial for Spatial Transformer Networks. https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html#sphx-glr-intermediate-spatial-transformer-tutorial-py


