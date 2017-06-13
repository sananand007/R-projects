# R-projects
## Random and Statistically interesting R projects/assignments

### Some amazing R resources and Books that are going to help you out a lot

| Book     	    								| Link          																	| Author 				|
| ------------- 								|:---------------------------------------------------------------------------------:| ---------------------:|
| Hands-On Programming with R   				| C:\Users\sanan\OneDrive\dumps\R-books 											| Garrett Grolemund 	|
| Efficient R Programming      					| https://csgillespie.github.io/efficientR/      									| Colin Gillespie 		|
| Advanced R      								| http://adv-r.had.co.nz/      														| Hadley Wickham 		|
|Introduction to Data Science with R			| http://r4ds.had.co.nz/															| Garrett Grolemund		|	 
|												|https://www.safaribooksonline.com/library/view/introduction-to-data/9781491915028/	| Garrett Grolemund		|
|(tidyr/dplyr)Expert Data Wrangling with R		|https://www.rstudio.com/resources/webinars/data-wrangling-with-r-and-rstudio/		| Garrett Grolemund		|
| Data Camp										|https://www.datacamp.com/courses													|						|		

## Mostyl you will not need this , but you can install packages directly from github
	- devtools::install_github("rstudio/<package name>")

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
	- Caret Package - http://topepo.github.io/caret/
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
			+ **Coeffient of Determination** = what % of the total variation is not explained by the variation in x or by the regression Linear
				- formula of Ratio [not described by the line] = $$\frac{SE_{line}}{SE_{y}}$$
				- % of total variation is described by the variation in x = $$R^2 = 1 \m \frac{SE_{line}}{SE_{y}}$$
				- If the $$SE_{line}$$ is small , that means the line is a good fit , hence the $$R^2$$ will be close to 1 , hence Line is not a good fit
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
	- **Covariate Creation**
		- Using Covariates or featues
		- raw data can take the form of a image or a text file
		- Transforming tidy covariates
		- Raw Data depends heavily on applications
		- Balancing act is the summurization vs. information Loss
		- Text files 
			+ frequency of words
			+ frequency of phrases
			+ frequency of capital letters
		- Webpages
			+ A/B testing [types of images , position of elements]
		- Images
			+ Edges, corners, blobs, ridges [face detection mostly]
		- People
			+ Height, weight, color, sex and origin
		- Tidy Covariates
			+ More necessary on regression,svm's than for others ie, Classification trees
			+ Should be done **only on the training set**
			+ Generally we should spend more time on exploratory data analysis
			+ New Covariates should be added to data frames
			+ Turn covariates which are factor variables into dummy variables/indicator variables
				- Better to turn these qualitative variables into quantitative variables as it is difficult to implement any algorithms as such
			+ Removing zero covariates , since all the predictors are not necessary to be unique and useful 
			+ Curvy model fitting using **splines** package
			+ We would have to create the covariates on the test data set using the same procedure that was used on the training set 
			+ Deeplearning tutorial and creation of features : http://www.cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf
		- Level-1 feature creation : raw data to covariates
		- Level-2 feature creation : covariates to new covariates 
			+ using the caret package , preprocess() function: http://topepo.github.io/caret/pre-processing.html
		- Preprocessing using PCA	
			+ All operation and model building has to happen in the training set only
			+ The idea is to "capture as much information" as possible
			+ if you have multivariate variables X1, ... Xn 
				- Find a new set of multivariate variables that are uncorrelated and find as much variance as possible
				- if you put all the variables together in one matrix , find the best matrix created with fewer variables , ie . Lower rank that explain the whole of the data
	- **SVD - Singular value decomposition**
		+ wiki : https://en.wikipedia.org/wiki/Singular_value_decomposition
		+ Take a look at the MIT lecture : 
		+ Take a look at the pdf : https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf
		+ equation : **A = UDV*** , where A is a mxn real matrix, with m>n, then A can be written as a so called SVD 
			- U is a left singular vector ie, orthogonal
			- V is a right singular vector ie, orthogonal
			- D is a diaginal matrix
	- **PCA - Principal Component analysis**
		+ It can reduce the number of quantitative variables present
		+ Check PCA from **Elements of Statistical Learning**
		+ Some books
			- Modern applied statistics with S
			- introduction to statistical learning
	- **Splitting into Decision Trees**
		+ Interatively split variables into groups
		+ Basic Algorithm	of decision trees - https://en.wikipedia.org/wiki/Decision_tree_learning
			- Start with Variables in one groups
			- Find the variable/split that best describes the outcomes 
			- Divide the data into two groups ("leaves") on that split ("node")
			- Within each split , find the best variable/split that separates the outcomes
			- continue till the groups are sufficiently pure or homogenous
			- Classification trees are non-linear models 
			- They use interactions between variables
			- There are multiple tree building options in R , such as : rpart , party, tree -> all are tree packages 
			- Book : https://www.amazon.com/Classification-Regression-Trees-Leo-Breiman/dp/0412048418
	- **Bagging** - Bootstrapping with aggregated values
		- resample your data , Refit your non-linear model, average those modelfilts together over resamples to get a smoother model fit
		- Take a look at the below functions thouroughly and understand how these functions are written
			+ ctreeBag$fit
			+ ctreeBag$pred
			+ ctreeBag$aggregate
		Some resources to look into bagging
			+ https://en.wikipedia.org/wiki/Bootstrap_aggregating
			+ http://stat.ethz.ch/education/semesters/FS_2008/CompStat/sk-ch8.pdf
	- **Random Forest**
		- Bootstrap Samples 
		- At each split, bootstrap variables
		- Grow multiple trees and vote
		- Some resources to take a look from in details
			+ https://en.wikipedia.org/wiki/Random_forest
			+ http://www.robots.ox.ac.uk/~az/lectures/ml/
			+ 
	- **Boosting**
		- Take a number of possibly weak predictors 
		- Weigh them and add them up
		- Add them up and get a stronger predictor
		- R's Multiple boosing libraries to be used and learned
			+ gbm, mboost, ada, gamboost
		- Some resources
			+ http://webee.technion.ac.il/people/rmeir/BoostingTutorial.pdf
			+ https://en.wikipedia.org/wiki/Gradient_boosting
			+ Boosting tutorial : http://www.cc.gatech.edu/~thad/6601-gradAI-fall2013/boosting.pdf
			+ http://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf
			+ https://kaggle2.blob.core.windows.net/wiki-files/327/09ccf652-8c1c-4a3d-b979-ce2369c985e4/Willem%20Mestrom%20-%20Milestone%201%20Description%20V2%202.pdf
	- **Model Based Prediction**
		- Linear discrimant analysis assumes the function is a multivariate gaussian
		- Quadratic discrimant analysis assumes that the function is a multivariate gaussian with different covariances
		- Model Based Prediction : assumes more complicated versions of the covariance matrix
		- Bayes Theorem & Naive Bayes (useful, if you have a large numeber of features)
		- References present here: 
			+ http://www.stat.washington.edu/mclust/
			+ http://statweb.stanford.edu/~tibs/ElemStatLearn/
			+ https://en.wikipedia.org/wiki/Linear_discriminant_analysis
			+ https://en.wikipedia.org/wiki/Quadratic_classifier
	- Regularized Regression
		+ As the number of predictors increase the training set erros always goes down or decrese
		+ But after a certain time, due to overfitting the number the training set error will start to decrease
	- Decomposing expected Prediction error
		+ Irreducible error + Bias^2 + Variance
	- Hard thresholding
	- regularized regression
	- Ridge regression path
	- Important to pick up the correct lambda variable that is optimum for picking up bias for variance
	- Some other Links are such that : 
		+ http://www.cbcb.umd.edu/~hcorrada/PracticalML/
		+ http://www.biostat.jhsph.edu/~ririzarr/Teaching/649/
		+ http://www.cbcb.umd.edu/~hcorrada/AMSC689.html#readings
		+ **Caret Package** has some other packages that can be used here, to fit different kind of prediction methods
			- ridge
			- lasso
			- relaxo
	- Combining Predictors
		- Basic intuition is voting, just like in the case of Random forests 
		- Even simple Blending can be useful
		- Build an odd number of models
		- Predict with each model
		- Predict the class by majority vote
	- Forcasting
		+ Data are dependent over time
		+ Specific pattern type
		+ Beware of extrapolation times and heat maps also, not necessary that they will convey the correct information
		+ Time series usage and decomposition 
			- Trend - Consistently increases over time
			- Seasonal - Where there is a pattern over a fixed period of time
			- Cyclic - When data rises and falls over non-fixed periods
		+ Break into Train and test sets 
		+ Use simple movging average
		+ Exponential smoothing
		+ Use accuracy to get the forcast
		+ Some Information and links that are useful to look 
			+ https://www.otexts.org/fpp/6/1
			+ Forcasting : https://www.otexts.org/fpp
			+ https://en.wikipedia.org/wiki/Forecasting - check the github for code as well
	- Unsupervised Learning
		+ https://en.wikipedia.org/wiki/Recommender_system
		+ cl_predict function in the clue package


## Machine Learning Quiz 4
	
	-	Fit (1) a random forest predictor relating the factor variable y to the remaining variables and (2) a boosted predictor using the "gbm" method. Fit these both with the train() command in the caret package.
		What are the accuracies for the two approaches on the test data set? What is the accuracy among the test set samples where the two methods agree?
	-  	Set the seed to 62433 and predict diagnosis with all the other variables using a random forest ("rf"), 
		boosted trees ("gbm") and linear discriminant analysis ("lda") model. Stack the predictions together using random forests ("rf"). 
		What is the resulting accuracy on the test set? Is it better or worse than each of the individual predictions?
	-   Set the seed to 233 and fit a lasso model to predict Compressive Strength. Which variable is the last coefficient to be set to zero as the penalty increases? 
	-   Fit a model using the bats() function in the forecast package to the training time series. 
		Then forecast this model for the remaining time points. 
		For how many of the testing points is the true value within the 95% prediction interval bounds?
	-   Set the seed to 325 and fit a support vector machine using the e1071 package to predict Compressive Strength using the default settings. 
		Predict on the testing set. What is the RMSE?

## Machine Learning Project

	+	The goal of your project is to predict the manner in which they did the exercise. 
		This is the "classe" variable in the training set. You may use any of the other variables to predict with. 
		You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, 
		and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.		
	+	I used Random forests and gradient boosting machines for this purpose 
	+	Read Elements of Statistical Learning , to know whick Algorithm can perform better with Noise and why ?
	+ 	For my dataset , It seems the OOSE for Random_forest is lower than GBM

13. **Developing data products**
	+ 	Here we are very focussed on create R packages using R products like Shiny, rCharts, manipulate and googleVis
	+	A Data product is the producion output of a statistical analysis
	+ 	Book: Developing Data Products in R [This book can be got for free]
		This book introduces the topic of Developing Data Products in R. A data product is the ideal output of a Data Science experiment. This book is based on the Coursera Class “Developing Data Products” as part of the Data Science Specialization. Particular emphasis is paid to developing Shiny apps and interactive graphics.
		The book is available here: https://leanpub.com/ddp
		https://github.com/seankross/slides/tree/gh-pages/Developing_Data_Products
	+	Dr. Brian Caffo : https://sites.google.com/view/bcaffo/home	
	+	https://datasciencespecialization.github.io/Developing_Data_Products/welcome.html

14. **Shiny**
	+	Shiny Application and a shiny server
	+	Shiny server documentation : http://docs.rstudio.com/shiny-server/
		-	We will need to host the shiny code on a server 
		-	It will call R in the background to run your algorithm
		-	R studio has a free hosting server
		-	can user aws or shiny's own paid service
		-	Webserver that anyone can use **shinyapps.io**
	+	For Shiny server, we would need to install a Linux OS - Check how to do that
	+ 	Shiny apps .io project - https://datasciencespecialization.github.io/Developing_Data_Products/shinyproject.html
	+ 	Shiny is a web developing platform for R
	+ 	A little bit of knowlege of HTML , CSS and Java script is required for creating web products
	+ 	Shiny R tutorial :  https://shiny.rstudio.com/tutorial
	+	Main Steps to go through the shiny app
		- Begin each app with a template
		- Add elements as arguments to fluidPage()
		- Create reactive inputs with an *Input() function
		- Display reactive results with an *Output() function
		- use the server function to assemble inputs and outputs
	+ use the render() function to build reactive output to display the UI
	+ Reactivity is the only thing that is very important here, and that will be taken care by the shinyapp
	+ Server
		- Use the server function to assemble inputs into outputs
		1. Save the output that you build to output$
		2. Build the Output with a render*() function
		3. Access input values with input$
		- Create reactivity by using Inputs to build rendered Ouputs
	+ One directory with every file the app needs :
		- The shiny server will look into the directory for **app.R**, use only app.R
		- Put all the datasets that you need , datasets , images , cssm helper scripts etc..
		- If your app takes a lot of code for ui and server , it would be beneficial to take up two codes **ui.R** and **server.R** , use only these names
		- Shinyapps.io helps you run your app when you want to, it is a sever maintained by R studio
	+ Sharing
		- Save your app in its own directory as app.R, or ui.R and server.R
		- Host apps at shinyapps.io by:
			1. Sign up a free shinyapps.io account
			2. Install the shinyapps package
			3. Build your Own server with Shiny Server or Shiny Server Pro
	+ Reactivity
		- Reactive functions are functions that take reactive values and knows what to do with them
		- Example: renderPlot()
		- Use of Reactive values , These act as datastreams that flow through your app
		- Input list is a list if reactive values
		- you can only call a reactive value from a function that is designed to work with one
		- Toolkit:
			+ renderPlot( {
							hist(rnorm(input$number))
							}), 
			+ object will respond to any reactive value in the code
			+ renderPlot() just runs the whole block of code associated with it
			+ construct output using output$
		- Modulazing using reactive() function , This builds a reactive object
		- Reactive functions are basically functions
			+ It is very important that we put () for all the reactive expressions
			+ reactive expressions just make sure that they pass on the notifications
			+ reactive expressions know whether they are valid or invalid
			+ reactive expressions cache their values (the expression will return the most recent value, unless it has become invalidated)
				this results in less computation
			+ reactive() makes an object to use 
			+ reactive expressions are themselves reactive
			+ call a reactive expression like a function
		- render*() functions make objects to display , always save the output to output$ , It makes an observer object that has a block of code 
			associated with it
		- isolate() - This creates output that is not reactive
		- observeEvent() - Triggers code to run by the server
		- observe()
		- Delay reactions with eventReactive()
		- eventReactive() 
			+ Is used solely for delaying reactions, while usind a actionButton()
		- What is the difference between observeEvent() and eventreactive() in a program ?
			+ https://shiny.rstudio.com/articles/action-buttons.html
		- using observeEvent() we can override the data object to have any other form of data
			+ overriding can only be possible using reactiveValues() with multiple input types
			+ reactiveValues create a list of reactive values to manipulate programitically using observeEvent()
		- Reduce repitition - place code where it will be re-run as little as necessary
		- Code inside the server function will be run once per end user as multiple copies of the same code is created 
		- shiny showcase : https://www.rstudio.com/products/shiny/shiny-user-showcase/
		- shiny gallery : https://shiny.rstudio.com/gallery/
		- shiny comes with a list of tags, tags$h1() to add html functionality, or we can do names(tags)
		- shiny gadget will have to have the ui part and the server part
		- Static Elements : tags$
			+ tags$a() -> a tag is important as the hyperlink with the href argument
			+ tags$p() -> paragraph
			+ tags$em() -> emphasis/Italics
			+ tags$strong() -> bold
			+ tags$code() -> styling
		- Adding Images : To add an image from a file, save the file in a subdirectory names **www**
		- We can also pass the html directly to the fluidPage()
		- Layout Functions
			+ Use fluidrow() and column() to shift and offset images and elements and arrange in rows and columns
			+ Use different type of *panel() to do something different within your layout
			+ sidebarLayout() - use sidebarPanel() and mainPanel() to divide app into two sections
			+ navbarPage() - instead of tabsetPanel() we can also use this
			+ navbarMenu() combines tab links into a dropdown menu for navbarPage()
		- check out Dynamic Dashboards with shiny webinar , to know more about shiny dashboards
		- The most popular HTML, CSS, and JavaScript framework for developing responsive, mobile first projects on the web. 
			+ check bootstrap 4 CSS framework , http://getbootstrap.com
		- Free codeacademy tutorial on CSS and HTML http://www.codeacademy.com/tracks/web
		- Use jQuery to add javascript 
		- more tutorials : http://zevross.com/blog/2016/04/19/r-powered-web-applications-with-shiny-a-tutorial-and-cheat-sheet-with-40-example-apps/		
	+ Hans Rosling - The first time show of moving charts 
		- google vis : https://cran.r-project.org/web/packages/googleVis/googleVis.pdf
		- Use the DataCamp tutorial to work on Google vis
14. **Plotly**
		+ For using plotly , the data should be in the form of a giant single data frame
		+ Plotly can also give graphs for multi-line stock market graph that can be put in the webbage
		+ Plotly can also help with heatmap, boxplots , histgrams , surface for 3D etc ..
		+ plotly account - with github 
		+ plotly tutorials with R : https://plot.ly/r/ | https://cpsievert.github.io/plotly_book/ 
15. **Markdown**
		+ https://seankross.github.io/slides/<>.html - This is the repo slide present -> create gh-pages , repository
		+ github pages : https://pages.github.com/
16. **Leaflet**
		+ Javascript - widget, without knowing any JavaScript
17. **Analysis of IMDB Kaggle Dataset**
		+ Finding trends in budget of the movie and the imdb scores
		+ Finding trend in number of movies released belonging to a particular genre
	
	
	
	