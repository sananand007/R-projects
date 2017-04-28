# R-projects
## Random and Statistically interesting R projects/assignments

## Projects to be synced 

1. Statistical Inference project - 1
	+ Compare a random exponential distribution to a Normal Distribution
	+ Prove that a particular distribution is a Normal Distribution 
	+ Compare the distribution of 1000 random uniforms


2. Statistical Inference project - 2
	+ Analyse the ToothGrowth data 
	+ Do a 	exploratory Data analysis
	+ State conclusions to the fullest
	
3. **Added a Small tool to check MSE using R and Manipulate , with the Help of Coursera DS specialization**
	+ Using the Manipulate library to create the slider with R
	+ Using ggplot2 to plot a histogram of the child heights data from the galton data frame
	+ Using Manipulate to calculate the MSE by calculation of  squared vertical distances for Parent's distribution using y = x(Beta), where Beta is the slope of the equation
	+ Also demonstating what is linear fitting
	
4. **Adding A descrpiption Linear model Description and the use of the plot(lmfit) function**
	+ Plots showing the residual vs Fitted points | Q-Q normal plots , showing the change (as modeled against) setting + effort
	
5. **Multivariate Regression and Regression through the origin**
	+ Basic Multivariate Regression models
	+ Regression threough the Origin 
	+ Using datasets such as swiss, Insectsprays 
	+ Understanding the effect of multiple predictors/covariates on response and the other variates 
	+ Factor variables in a linear model

6. **Adjustment**
	+ Idea of putting a regressor through a linear model to investigate the role of a third variable on the relationship between the other two
	+ Case:1 - Strong marginal effect (ie, change in the intercept), when x or the regressor is discarded  and very less effect when x is included 
	+ Case:2 - less marginal effect (ie, change in the intercept), when x or the regressor is discarded  and very high effect when x is included 
	+ Case:3 - x is such that there is no overlap in the above two cases , and there will be cases when there will be ovelap occuring
	+ Case:4 - x is such that there is no marginal effect when we ignore x  , and there is a huge effect when we adjusted for x and we include x
	+ Case:5 - For this case there is no common slope, case where there are different intercepts and different slopes- For this case there is no Treatment effect and the treatment effect depends on what level of x we are at .
		- You will need a interaction term in the model , for this case , and also common slopes will not be there
	+ More the number of variables , it is better to do residual analysis in that case 	
	+ **Simpsons Paradox** - Simpson’s paradox refers to a phenomena whereby the association between a pair of variables
		(X, Y ) reverses sign upon conditioning of a third variable, Z, regardless of the value taken
		by Z. 
		- You look at a variable as it relates to an outcome and that affect reverses itself with the inclusion of another variable (regressor)
		- Things can change to the exact opposite if you perform adjustment
		### Influential measures and influential outliers , Influence.measures | Residuals , Diagnostics & Variations
			- When a sample is included in a model, it pulls the regression line closer to itself (orange line) than that of the model which excludes it (black line.) 
			- Its residual, the difference between its actual y value and that of a regression line, is thus smaller in magnitude when it is included (orange dots) than when it is omitted (black dots.) 
			- The ratio of these two residuals, orange to black, is therefore small in magnitude for an influential sample. For a sample which is not influential the ratio would be close to 1. 
			- Hence, 1 minus the ratio is a measure of influence, near 0 for points which are not influential, and near 1 for points which are. This measure is called **influence or leverage or hatvalue**
			- **rstandard**: The function, rstandard, computes the standardized residual
			-**rstudent**: The function, rstudent, calculates Studentized residuals for each sample
			-**Cook's distance** is the last influence measure we will consider. It is essentially the sum of squared differences between values fitted with and without a particular sample. 
			- It is normalized (divided by) residual sample variance times the number of predictors which is 2 in our case (the intercept and x.)
			- It essentially tells how much a given sample changes a model
			- **cooks.distance**: The function, cooks.distance, will calculate Cook's distance for each sample
7. **P-value & Hypothesis Testing**
	+ Null Hypothesis is always the case you say that a Response is not dependent on a Predictor, hence you consider the case when the p-value is low or otherwise lower than 0.05, that says that the predictor/Regressor is significant and there is enough evidence that the Null hypothesis can be rejected

8. **GLM - Generalized Lnear Models**
	• An exponential family model for the response 
	• A systematic component via a linear predictor
	• A link function that connects the means of the response to the linear predictor

	### Other properties : 
	- Used Generalized Estimation equation 
	- Predictors are linear but link function is non linear
	- Allows response variables to take any form of exponential distribution
	- Handles non-normality effectively (logit, probit/inverse Logit, LogLinear)
	- Handles high correlation, which OLS and MLE fail to do
	- odds Ration = exp(b1)

9. **Variance Inflation**
	- variance inflation is due to correlated regressors, these can be gauged by calculating the standard error/variances
	-  theoretical estimates contain an unknown constant of proportionality. We therefore depend on ratios of theoretical estimates called Variance Inflation Factors, or VIFs.
	-	A variance inflation factor (VIF) is a ratio of estimated variances, 
		the variance due to including the ith regressor, divided by that due to including a corresponding ideal regressor which is uncorrelated with the others	
	- VIF is the square of standard error inflation.
	- omitting a correlated regressor can bias estimates of a coefficient
	- As the number of regressors approaches the number of data points (47), the residual sum of squares, also known as the deviance, approaches 0
	- An F statistic is a ratio of two sums of squares divided by their respective degrees of freedom
	- R's function, deviance(model), calculates the residual sum of squares, also known as the deviance, of the linear model given as its argument
	- **F-Test [ANOVA]** , $$F = \fraction{variation between sample means}{ variation within the samples}$$
	- Linear regression minimizes the squared difference between predicted and actual observations, i.e., minimizes the variance of the residual.
		
10. **Regression Models-Project**
	- This project is about analysis of the mtcars dataset and find out the relation between the response MPG against the various covariates present 
	- Use different model fits like linear regression and Multivariate regression to find out the coefficients of the model ie, fitted
	- Find out the p-value to weather we can reject the Null Hypothesis that a particular covariate is insignificant
	- Finding R^2 to check the "goodness of fit" and seeing how much of the variance we can explain here
	- Finding the Autocorellation function graph as well to check the residuals Vs. Fitted values