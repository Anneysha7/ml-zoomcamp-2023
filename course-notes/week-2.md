# Week 2

### Why do we apply a logarithmic transformation to a variable during EDA?
Logarithms transform huge differences in numbers into comparable differences, while still maintaining the extent of the differences.  

Let us take an example: As seen in the video, we have a few cars that cost more than $80,000. However, most of our cars cost much less. Since we have such a huge range of car prices, the average cost of cars (~$25,000 - described in the video) is very close to 0, which should not be the case.

Therefore, we use logarithmic transformations to get rid of long-tails (skewness) of our plot as much as possible.

Here is the mathematical demonstration:  

$log(x^n) = n.logx$
<br>

When 1,000,000 is transformed using logarithm:  
$log(10^6) = 6.log10$  
= 6 (base 10)

Transforming 1 similarly:  
$log(1) = 0$  

Therefore, our range shrinks from 1,000,000 - 1 to 6 - 1.

$log(0)$ is not defined. Therefore, when applying logarithmic transformations, our starting point needs to be 1.

To work around this issue, we simply add 1 to every value that we might have. This works fine because we are looking at the relative differences in the range to train our ML model, in the first place. 

Therefore, we use `np.log1p()` instead of `np.log()`. Using `np.log()` would give us a $log(0)$ error.


### Train, Validate, Test (60:20:20)
- Set seed to ensure that the shuffling can be reproduced on any computer: `np.random.seed(1)`.
- Shuffle the dataset to ensure that each of train, validate and test datasets is well-distributed. Shuffle index by getting an array of indices `np.arange(n)` and then shuffle index using `np.random.shuffle(idx)`.
- Split dataset into 60:20:20 ratio. Use `df.iloc[start:end]`.
- Get rid of the indices for all the datasets using `df_train.reset_index(drop=True, inplace=True)`. `inplace` makes changes permanent, otherwise changes do not apply to future blocks.
- Apply logarithmic transformation.
- **Important!** We are using a numpy array for our machine learning. Delete the specific column from train, test and val columns to prevent any confusion.

### Linear regression model: y = mx + nx + ox + c
We can train on multiple columns (called features)! Training on multiple features helps avoid underfitting. However, we need to be careful that training too many features might lead to overfitting.  
This will be covered below.

**Implementing linear regression from scratch**  
Simply implement the equation for y. The linear regression function takes a numpy array as input, iterates over length of the array `n` and performs the equation for all values in the array. It returns the prediction `pred` which is initialised as the intercept `c` before running the linear regression equation.

However, we can also use pre-written functions by using the `sklearn.model_selection` module, and the `test_train_split(arrays, test_size, train_size, random_state=(seed), shuffle=True(default))` function. If we use this function, then we would not even need to shuffle or split our dataset in the first place.

Essentially, we use matrix multiplication to make our linear_regression function faster. A matrix consists multiple rows of features for each model, which is multiplied with a rectangular matrix.

Xw = Y
X is the feature matrix.
Y is the prediction matrix.
w is a rectangular matrix (1xn).

w = (X''X)'. X''Y, where X'' is the transformation of matrix X. This can be obtained by using `X.T` in `numpy`.

This w matrix provides us the formula required to reach from our feature matrix X to the prediction matrix Y.

Machine Learning is purely Mathematics! 


