# RBM
Photo classifier: uses Gaussian-Bernoulli-RBM to extract features and classify by SVM

### Brief
```
Implementation of a Gaussian-Bernoulli-RBM in order to extract features from Images and classfiy by SVM from scikit-learn.
1.the system takes an existing image dataset and convert pixels to data.
2.devide data randomly to a training and test sets
3.runs an RBM to extract features from images data.
4.classifieds by SVM.
System has a GUI bulit in TKINTER.
Creates temporary folder for the training and the test set.



It includes a terminal user interface which allows to evaluate and control 
the input and the output, which includes login credentials (username, password), 
credit info (card number, security code) and amount of order.

DISCLAIMER:
This project is non-profit and is intended to serve for educational purposes only.
It is not meant to infringe copyright rights by any means.
Please notify the repository owner of any infringements and they will be removed.
```
### Research Papers
Learning Features for Tissue Classifcation
with the Classifcation Restricted
Boltzmann Machine
Gijs van Tulder and Marleen de Bruijne

### Installing and Running
- Pytorch
- Clone the Project
```
git clone https://github.com/IraGoor/RBM
cd RBM
```
- Execute from Anaconda CLI
```
cd ..
conda init
conda activate base
python main.py
```
- Run Example

### Prerequisites and Libraries
- Spyder (IDE)
- Anaconda (Python3 Distribution)
- Numpy (Scientific Calculations)
- Pytorch (Tensor Calculations)
