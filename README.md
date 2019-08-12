# Parametric Majorization for Data-Driven Energy Minimization Methods


This is code for the paper titled "Parametric Majorization for Data-Driven Energy Minimization Methods". The experiments in section 4.2 and 4.3 are implemented in PyTorch, whereas the experiment in 4.1 is implemented in Matlab [Code is being prepared right now].


## INSTALLATION:

1) Have conda installed
2) Run ```conda env create -f environment.yml``` to recreate the exact environment.
3) Download the datasets. You need BSDS300 for the denoising example and cityscapes for the segmentation example.
- ```BSDS``` can be found at https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- ```cityscapes``` can be found at https://www.cityscapes-dataset.com/

#### Dependencies:

While ```environment.yml``` contains the exact package versions to reproduce this work, ```environment_minimal.yml``` contains the general list of dependencies.

## USAGE:

The jupyter notebooks should give you a reasonable overview on how to use this framework.
As a minimum working example for the training of denoising filters, consider the following code snippet
```
import bilevelsurrogates as Sur

# We assume data is given as a torch.utils.data.Dataset
samples = Sur.data.Samples(datasetTrain, batch_size, device=device, dtype=dtype)

# Define a model:
dictionary = Sur.DCTConvolution(in_channels=1, out_channels=48, kernel_size=7)
energy = Sur.model.AnalysisSparsity(dictionary)
loss = Sur.loss.PSNR()

# Define parameters
algorithm='joint-dual'
iterative_setup = Sur.training.default_setup('IterativeLearning', algorithm)
training_setup = Sur.training.default_setup('DiscriminativeLearning', algorithm)

# Train
subroutine = Sur.training.DiscriminativeLearning(energy, loss, samples, training_setup, algorithm=algorithm)
optimizer =  Sur.training.IterativeLearning(subroutine, iterative_setup)
optimizer.run();

# Visualize results
Sur.visualize(energy.operator);
```


## CITATION:

If you use this framework in your work, please cite the paper

**Parametric Majorization for Data-Driven Energy Minimization Methods**
    (J. Geiping, M. Moeller),
    To appear in IEEE International Conference on Computer Vision (ICCV), 2019.


## SPECIAL THANKS

https://github.com/Luolc/AdaBound

https://github.com/meetshah1995/pytorch-semseg/

https://github.com/fvisin/dataset_loaders

## CONTACT
If you have any questions or comments, just open an issue.
Or write an email to ```jonas.geiping at uni-siegen.de``` or ```michael.moeller at uni-siegen.de ```.
