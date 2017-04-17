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