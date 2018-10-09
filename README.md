# Neural network-based videogame for Dysgraphia classification

- Repository GitHub (NN):
https://github.com/vincenzorin/project_ia
- Unity Repository (Android interface):
https://developer.cloud.unity3d.com/orgs/nemesis92/projects/ia_for_dysgraphia/
- YouTube Demo: 
https://youtu.be/WurHp_ktDmo

[![Video](http://i3.ytimg.com/vi/WurHp_ktDmo/maxresdefault.jpg)](http://i3.ytimg.com/vi/WurHp_ktDmo/maxresdefault.jpg)


This project is composed of three elements: 
1. The dataset builder script, written in Python, which mines elements of HTML pages requested via HTTP. 
2. The user interface, an Android game developed using Unity and $N Multistroke Recognizer libraries in C++ 
3. A Neural Network, designed using PyTorch libraries.  

The application developed exploits gamification to submit exercises to the user, which are collected and analysed to serve as input for the Recurrent Neural Network designed.  This RNN takes into account a set of features as slope, execution time and quality of the handwritten text in order to classify the input. 

## RESULTS
The RNN has achieved [Loss 0.012], [Accuracy: 0.95] while training.
