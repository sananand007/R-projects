# Regression Models

# Plotting The MSE using ggplot2

library(ggplot2);library(manipulate);library(UsingR)

data(galton)
myHist <- function(mu) {
  mse <- mean((galton$child - mu)^2)
  g<-ggplot(galton, aes(x=child)) +
      geom_histogram(fill="lightblue", color="black", binwidth = 1) +
      geom_vline(xintercept = mu, size=4) +
      ggtitle(paste("mu =", mu, "MSE =", round(mse,2), sep = " "))
  g
}
manipulate(myHist(mu), mu=slider(62,74, step = 0.5))