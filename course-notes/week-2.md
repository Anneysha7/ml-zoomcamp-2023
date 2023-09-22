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
