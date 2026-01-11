- you should comment the code better. In particular you should document the purpose of classes and functions, their parameters, return values, and any important side effects.
- I understand that you do not perdorm minibatches. However, you should at least mention this design choice in the documentation, so that users are aware of it.
- I don't think the algorithm is enirely correct. The shuffllying of each data set should be made at every epoch, not only once at the beginning. Otherwise, the model may overfit to the order of the data.
- I don't see the test data. You shoudl use the test data to verify model performance suring training.
- Take the habit of using namespaces in your code.

## Final comments

Overall, the code is well structured and implements a basic neural network training pipeline. However, there are several areas where improvements can be made, particularly in documentation, handling of data shuffling, and testing. Addressing these issues will enhance the usability and reliability of the code.

In a more professional implementation I suggest to use an external library for automatic differentiation. However, I appreciate the effort of implementing it from scratch for learning purposes.

