# Adversarial Generator for MNIST

This is an end to end example of using adversarial examples to exploit a deep convolutional MNIST classifier. It uses the wiggling method as described in [Andrej Karpathy's blog post](http://karpathy.github.io/2015/03/30/breaking-convnets/)

## Instructions
1. Run `train.py` to get a trained mnist classifier using the deep convolutional network described [here](https://www.tensorflow.org/get_started/mnist/pros#deep-mnist-for-experts). By default, the saved model will be trained on MNIST for 1000 steps, save MNIST data to `tmp/data` and write its `model.meta` file to `tmp/run`.

2. Run `adversarial.py`. By default, this will generate 10 adversarial examples for MNIST "2" samples which are classified by the trained network as "6", create an image `output.jpg` containing 3 columns (original, delta, adversarial example) and rows being each of the 10 examples.

# Parameters
## train.py
- `--data_dir`: The data directory to save/load MNIST data.
- `--output_dir`: The output directory for the trained model.
- `--train_steps`: The amount of steps to train. (Observation, models which are triande with more steps take longer to generate adversarial examples for!)

## adversarial.py
- `--origin`: The origin MNIST class to generate adversarial examples for.
- `--target`: The target MNIST class to pertubate origin samples into.
- `--output`: The desired filename of the output table image. It contains 3 columns (original, delta, adversarial example) and sample_size number of rows with each row being a generated example.
- `--eps`: The epsilon amount to wiggle towards the network gradient of target class.
- `--wiggle_steps`: Upper bound on the number of wiggle operations in case epsilon is too big.
- `--sample_size`: The number of samples to generate. (origin.sample_size = target.sample_size) 
- `--model_file`: The filename of the saved meta graph.
- `--model_dir`: The model directory to load the trained MNIST classifier.
- `--data_dir`: The data directory to load MNIST data.
- `--verbose`: Turn this on to see logging of wiggling operations. Useful for monitoring any over stepping.
