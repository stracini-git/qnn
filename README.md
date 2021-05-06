# Quantization Schemes For Training Neural Networks

This work investigates the possibility of directly training quantized neural networks. 
First, a weight quantization scheme is proposed which shows that little to no performance 
loss is incurred when quantizing weights down to 4 bits. Second, it investigates the impact 
of a similar quantization scheme on the activations of neurons. Experiments performed with 
a simple fully connected network on the MNIST dataset as well as with a residual neural 
network on the CIFAR-10 dataset show no significant loss when training with 4 bit weights 
and activations. 

## Requirements
- the **MNIST**/**CIFAR10** datasets are automatically downloaded form the TF repo when they are first used


## Run Experiment
By default it runs LeNet300 using 32 bits for weights and activations:

```markdown
python Trainer.py 
```

**--wbits** specifies the number of bits for weights\
**--abits** specifies the number of bits for activations

```markdown
python Trainer.py --wbits 4 --abits 4
```
