#Regression Models

#Plotting The MSE using ggplot2

#Plotting the child vs Father , paired pairs for father and child

library(ggplot2);library(manipulate);library(UsingR)

data(galton)

# myHist <- function(mu) 
# {
#   mse <- mean((galton$child - mu)^2)
#   g<-ggplot(galton, aes(x=child)) +
#       geom_histogram(fill="lightblue", color="black", binwidth = 1) +
#       geom_vline(xintercept = mu, size=4) +
#       ggtitle(paste("mu =", mu, "MSE =", round(mse,2), sep = " "))
#   g
# }
#manipulate(myHist(mu), mu=slider(62,74, step = 0.5))

# Getting the Mean Squared Errors using a Line
library(dplyr)

y<-galton$child - mean(galton$child)
x<-galton$parent - mean(galton$parent)
n<-galton$child
m<-galton$parent

freq.data<- as.data.frame(table(x,y))
#freq.data<- as.data.frame(table(n,m))

names(freq.data)<-c('child', 'parent', 'freq')

freq.data$child<-as.numeric(as.character(freq.data$child))
freq.data$parent<-as.numeric(as.character(freq.data$parent))

myplot<- function(beta){
  g<-ggplot(filter(freq.data, freq>0), aes(x=parent, y=child))
  g<-g+ scale_size(range = c(2,20), guide = "none")
  g<-g+ geom_point(color="grey40", aes(size=freq+20, show_guide=FALSE))
  g<-g+ geom_point(aes(colour=freq, size=freq))
  g<-g+ scale_colour_gradient(low = "lightblue", high = "white")
  g<-g+ geom_abline(intercept = 0, slope = beta, size=3)
  mse<-mean((y-beta*x)^2)
  g<-g+ggtitle(paste("beta=", beta, "mse=", round(mse,3)))
g
}

manipulate(myplot(beta), beta=slider(0.6, 1.2, step=0.2))

