# Geometric distribution

## On examples from quant interview problems

## Reminder 

$$P(X = success) = p$$
$$P(X = fail) = 1 - p$$

$$ \mu = \frac{1}{p}$$

$$ \sigma^2 = \frac{1-p}{p^2} $$

## Task 1: You are flipping a fair coin repeatedly. How many times on average do you need to flip it to get 5 heads in a row?

To find out how many times on average you need to flip a fair coin to get $n$ heads in a row, 
we can use the concept of Markov chain and induction approach.

Let $E[f(n)]$ be the expected number of coin tosses to get $n$ heads in a row. 
Using the Markov chain approach, we divide the experiment into states , state 
$i$ defines we have seen i heads in a row. We know state before 
(n+1) heads in a row must be 
$n$ heads in a row. Conditioned on state
$n$ heads in a row, there is a $1/2$  probability it will go to 
$n+1$ heads in a row (the new toss yields H) and the process stops. There is also a 
$1/2$ probability that it will go to the starting state 0 (the new toss yields T) and we need another expecct
$E[f(n+1)]$ tosses to reach 
$(n+1)$ heads in a row ( i.e. In addition to the existing 
$E[f(n)]$ steps to reach the state of $n$ heads). So we have

$$E[f(n+1)]= (1+E[f(n)]) + (1+E[f(n)]+E[f(n+1)]) $$

This simplifies to $$E[f(n+1)]=2E[f(n)]+2$$. 
This recursion can be solved easily and yields E[f(n)]=2 n+1 −2. Therefore, on average, you need to flip a fair coi $2^{n+1} −$2 times to get 5 heads in a row.

## Task 2: 
On average, how many times must a 6-sided die be rolled until the sequence of 
65 (i.e. a 6 followed by a 5) appears?

There are three possibilities once we roll a 
6: (a) we roll a 5, 
(b) we roll a 6, or 
(c) we start all over again.

Let E be the expected number of rolls until 65 and let E6 be the expected number of rolls until 
65 when we start with a rolled 6. Then, we can set up two linear equations based on these expectations.

$$E = 1 + 1/6 E6 + 5/6 E$$

$$E6 = 1 + 1/6 E6 + 4/6 E$$

$$E = 6 + E6, E6 = 30$$

## Task 3: Let $X_i$ be IID random variables with uniform distribution between 0 and 1. What are the cumulative distribution function, the probability density function and expected value of 
$Z_n=max(X1, X2, ...,Xn)$? What are the cumulative distribution function, the probability density function and expected value of $Y_n=min(X 1,X2,...,X_n)$? 


$$ P(Z_n \leq y) = F_{Z_n}(y) = y^n $$

So:

$$ p_{Z_n}(y) = (y^n)' = n y^{n-1} $$

And:

$$EX(Z_n) = \integral_0^1 y p_{Z_n}(y) dy = \integral_0^1 (n y^{n}) dy = \frac{n}{n+1}$$

$$ P(Y_n \geq y) = F_{Y_n}(y) = (1 - y)^n $$
So:
$$ p_{Y_n}(y) = (y^n)' = n (1 -y)^{n-1} $$

$$EX(Y_n) = \integral_0^1 y p_{Y_n}(y) dy = \integral_0^1 y n (1 -y)^{n-1} dy = 
n\integral_0^1 (1-x) x^{n-1} dx = n (1/n 1^n - 0 - 1/(n+1) 1^{n+1} + 0) = \frac{1}{n+1}$$


## Task 4: You are given a rooted tree with n nodes. On each step, you randomly choose a node and remove the subtree rooted by that node and the node itself; until all of them have been removed (that is, the root has been chosen). Find the expected number of steps in this proces. sot = size of tree

P(xi chosen) = 1/sot

P(xi will be removed) = depth / sot (to remove vertex needs to be an ancestor)

P(xi chosen | xi will be removed) = P(xi chosen and xi will be removed) / P(xi will be removed) = 1/sot * sot/depth = 1 / depth

In the end all vertices will be removed, so:

$$EX(xi chosen) = P(xi chosen | xi will be removed) = 1 / depth$$

So:

$$EX = \sum 1/depth$$

## Task 5: Is rolling a non-fair die has a greater probability of rolling doubles than rolling a fair die?

## Task 6: You are given a chessboard with 8x8 dimensions. Your friend asks you how many ways there are to place coins on the chessboard such that there are even number of coins in each row and in each column. Can you find it out?

## Task 7: 



