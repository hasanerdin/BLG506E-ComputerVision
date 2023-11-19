# HW1

K-Nearest Neighbors (k-NN)

 ![image](https://github.com/hasanerdin/HW1/assets/52705602/bba43c45-bfd2-4ede-977e-37694ca79002)
Figure 1: Representation of k-nn [1]

In this homework, we are asked to implement and learn k-NN algorithm. K-NN is a machine learning algorithm which is used for classification. 

Training Process: Whereas most of the other machine learning algorithms have a training process which includes feature extraction, data preprocessing and finding the best function to map input to output label, k-NN algorithm only stores all the training data and their labels in memory. Therefore, it is inefficient for memory consumption. Memory consumption is linearly increasing with the training data. However, it does not take any time for training. 

Test Process: In inference time, most of the other machine learning algorithms only make predictions, but, in k-NN it is required to travel all over the training data and calculate the distance between the data whose label we want to predict and each of the training data. According to these distances, we choose k training data which have smallest distance. In these k training data, we look at their labels and choose a label with majority voting. Also, in inference time we have computation complexity and still when training data increases, prediction time will increase, too.
	In the homework, we work on CIFAR-10 dataset but we test our code with subsampled dataset. Because it is faster way to understand every thing is working as good as we expected. 
	Until running k-nn algorithm on CIFAR-10 dataset, 
	We test distance computation functions which are two-loops, one-loop and vectorized. With help of the vectorization, we can run our code 450x faster than looping techniques. Before calculating the distances, we flatten the data to get vectors.
	Two-loop distance calculation: We writed two for loop to go over training and test datasets and calculate the distance between each pairs step by step. It takes too much time because of curse of algortihm complexity O(n.m) where n is the number of training data and m is the number of test data.
	 One-loop distance calculation: We only use one for loop to go over training data and calculate the distances between one of the training data and all test data vector. It is fater than two-loop calculation because its complexity is O(n) where n is the number of training data.
	Vectorized (no-loop) calculation: We use vectorization trick to run our function even more faster than previous implementations. 

〖(x-y)〗^2=x^2+ y^2-2*x*y				(1)

Firstly, we calculate the summation of the squared of the training data (x) and test data (y), respectively x2 and y2. We add additional axis to the result of the calculation made with training data, x2, which helps us to easly calculate x2 + y2. We use matrix multiplication to calculate x * y and find distances with equation (1).
	We learn how to choose a label for the test data in k-nn algorithm. Firstly, we need to calculate all the distances between test data and each training data. According to these distances, we choose k smallest distances. We find the labels of selected k nearest neighbours. Lastly, we make our prediction due to majority voting of the labels of k nearest neighbours. We changed k value and inspect how it affects the prediction process. We figure out than it is not always make our predictions better to choose larger k value. We need to run our algorithm and investigate the result. Visualization gives us better understanding of what is going on.
	We used topk function to find smallest k distances. We give transpose of the distance matrix because matrix is [number of training data, number of test data] but we want to get the k smallest distances of each test data as an output with [number of test data, k] shape.
	K smallest distances are sorted because it is wanted to select smallest label when the majority vote is even between different classes.
	To find majority voting in the sorted k smallest distance tensor, mode function is used. 
	We are working with small subset of the training set so that our success is low. We use all the training data for the training and do not make any accuracy calculation. To find which k value is best, we divide our training data into subsets. We select one of the subset as a validation set to test k hyperparameter while rest of the subsets are concatanated and used for distance calculation. This is known as cross-validation and it shows us that which k value is better than others on various subsets. At the end of the cross-validation process we choose k value with the best mean accuracy performance. Our performance increase from 27.8% to 28.2%.
	To divide training data into subsets, tensor_split function is used. Torch.chunk function is suggested in the homework but in the pytorch documentation it is indicated that when the input set is not divided by requested subfold number, torch.chunk function may not return requested number of subfold. It is recommended to use torch.tensor_split.
	For each k choices, we loop over all subfolds and calculate accuracy. Each subfold is selected as a validation data, orderly, and rest of them is used as a training data. KNNClassifier is build and accuracy is calculated with validation data with hyperparameter of k. Accuracy values are stored in a list which is stored in a dictionary with key k.
	After accuracies are calculated for each k choices with cross-validation, it is easy to find best k value according to the mean of the accuracies. 
	Until this point we only used a small portion of training data. But now we are sure that aour code is running excatly how we expected. When we use all the data as a training data and test the algortihm with test dataset, we get 33.86% accuracy.


References:
[1] K Neasrest Neighbours – Introduction to Machine Learning Algorithms, 21.10.2023, https://medium.com/@sachinsoni600517/k-nearest-neighbours-introduction-to-machine-learning-algorithms-9dbc9d9fb3b2

Recources:
https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
https://www.ibm.com/topics/knn
https://medium.com/@sachinsoni600517/k-nearest-neighbours-introduction-to-machine-learning-algorithms-9dbc9d9fb3b2

