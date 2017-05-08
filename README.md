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
		- **rstudent**: The function, rstudent, calculates Studentized residuals for each sample
		- **Cook's distance** is the last influence measure we will consider. It is essentially the sum of squared differences between values fitted with and without a particular sample. 
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
	
11. **Practical Machine Learning**
	- Uses in different fields like Biomedical and prediction 
	- Use of the R caret package
	- Presence of Materials for ML 
		+ The Elements of Statistical learning
		+ Stanford - ML course By Andrew Ng
		+ Kaggle Competitions
		+ https://www.quora.com/What-is-the-best-MOOC-to-get-started-in-Machine-Learning/answer/Xavier-Amatriain
		+ https://www.quora.com/Machine-Learning/How-do-I-learn-machine-learning-1
		+ http://www.sciencemag.org/site/feature/data/compsci/machine_learning.xhtml
		+ https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-867-machine-learning-fall-2006/lecture-notes/
		+ https://ocw.mit.edu/courses/find-by-topic/#cat=engineering&subcat=computerscience&spec=artificialintelligence
		+ http://www.stat.cmu.edu/~cshalizi/350/
		+https://www.kaggle.com/
		+ Notes on CMU - 36-705 | 10-715 | 
			- http://www.stat.cmu.edu/~larry/=stat705/
			- http://www.cs.cmu.edu/~epxing/Class/10701/lecture.html
			- Statistical ML [Larry Wesserman] - Spring 2016: Statistical Machine Learning (10-702/36-702)
	- Components of a Predictor
		+ question -> input data -> features -> algorithm -> parameters -> evaluation
	- Logistic Regression and Linear Kernel SVMs, PCA vs. Matrix Factorization, regularization, or gradient descent
	- In Sample vs Out of Sample
		+ In-Sample error - The error that you get on the same data that you used to build your predictor
		+ Out-of-Sample error - The error rate that you get on a new data set , Generalization error
		+ Insample error < out-of-sample-error
		+ Overfitting - Only matching your data very closely and will not match the other data that you have 
	- **Prediction study Design**
		+ Define your error rate
		+ Use Like Data to predict like	
		+ Split data into: 
			- Traning Set
			- Test Set
			- Validation Set
		+ On Training set pick features
			- Use cross-validation, What is cross-validation?
		+ If no validation
			- apply 1x to test set
		+ If validation
			- Apply to test set to refine and change the model to have a better fit
			- Get the test set error 
			- Apply 1x to the validation set  and that will give a good estimate of the out of sample error rate
		+ Hold one data set and keep it completely aside to run on your model to make sure how your model fits this and you can only apply you model on this data set only one time
		+ Avoid small sample sizes, so that we are not getting good prediction accuracy by chance
		+ Rules of Thumb
			- If there is a Large sample size
				+ 60% training
				+ 20% test
				+ 20% validation
			- If you have a medium sample size
				+ 60% training
				+ 40% test
			- If you have a small sample size
				+ Do cross-validation
				+ report the caveat of small size
			- Set aside test set and validation set and do not look at it when building a model, only 1 time this should be applied to your model at the end only
			- backtesting - mostly apply in finance kind of datasets , datasets consists of observations over time, split train/test in time chunks
			- All subsets should reflect as much diversity as possible
	- **Types of errors**
		- Postive = Identified | negative = rejected 
			+ True Positive = correctly identified
			+ False Positive = incorrectly identified
			+ True Negative = correctly rejected
			+ False Positive = incorrectly rejected
		- Statistical definitions : https://en.wikipedia.org/wiki/Sensitivity_and_specificity
			+ Sensitivity and specificity are statistical measures of the performance of a binary classification test, also known in statistics as classification function:
			+ Sensitivity (also called the true positive rate, the recall, or probability of detection[1] in some fields) measures the proportion of positives that are correctly identified as such (i.e. the percentage of sick people who are correctly identified as having the condition).
			+ Specificity (also called the true negative rate) measures the proportion of negatives that are correctly identified as such (i.e., the percentage of healthy people who are correctly identified as not having the condition).
		- Very important to know what population you are modelling from
		- Common Error Measure : For continous data we generally check the MSE and the RMSE, Median Absolute deviation
		- ROC curves - Receiver operating characteristic curves ie, you plot the Sensitivity Vs. Specificity
			+ Higher the area, better the predictor is
			+ The 45 degree line is  the point where we have the Probablity of 0.5
			+ The further you are to the upper left hand corner of the square/pltot, the better the ROC is, and the closer you are to the right hand corner of the plot 
			the , poorer the ROC is 
	- Cross Validation
		- Use the Training Set
		- Split it into training/test sets
		- Build the model on the training set
		- Evaluate on the test set
		- Repeat and average the estimated errors
		- Different types of cross validation techniques to estimate the Out of Sample accuracy/error rate
			+ Random Subsampling
				- without replacement
				- with replacement - bootstrap --> learnt this technique earlier, take a look into it
					- Underestimates the error , as some of the samples are considered more than once
					- Any of the models can be built by using the caret package in R
			+ K-fold 
				* Larger K = less bias, more variance
				* smaller K = more bias, less variance
			+ Leave one Out 
	- Caret Package
		- Cleaning
		- Data splitting
		- Training/Testing functions
		- Model Comparision [confusion matrix]
		- http://www.edii.uclm.es/~useR-2013/Tutorials/kuhn/user_caret_2up.pdf
		- https://cran.r-project.org/web/packages/caret/vignettes/caret.pdf
		- file:///C:/Users/sanan/Downloads/v28i05.pdf
		- Data Splicing 
		- time Splicing
		- Train options
			+ RMSE [Root mean square error]
			+ RSquared = R^2 - Coefficient of Determination, it suggers how close your model fits the original data
			+ Adjusted RSquared - The adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model
			+ Accuracy - Fraction correct
			+ Kappa - measure of concordance
			+ Usage of trainControl 
			+ Caret tutorial 
			+ Model training and tuning - http://topepo.github.io/caret/model-training-and-tuning.html
	- Machine Learning Algorithms [Try to simulate a ML Algorithm on your own that using Libraries]
		- Linear Discriminant analysis
		- regression
		- Naive Bayes
		- SVM [Support vector machines]
		- Classification to regression trees
		- Random Forests
		- Boosting
	- Plotting predictors
		- Make your plots only on the training set and not use the test set at all
		- Look for imbalance in outcomes/predictors 
		- Outliers
		- Groups of points not explained by the predictors 
		- Skewed variables
		- Standardization -> negating the mean and dividing by the standard deviation
		- Any formulas or values that we apply to the test set have to be got from the training set
		- Test-1
			
	