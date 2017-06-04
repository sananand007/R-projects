# demonstrating Plotly

library(plotly)
data("Exports")

head(Exports)
plot_ly(Exports, x=Exports$Country, y=Exports$Profit, mode="markers")

set.seed(2016-07-21)
temp<-rnorm(100, mean=30, sd=5)
pressue<-rnorm(100)
dttime<-1:100
plot_ly(x = temp, y=pressue, z=dttime, type = "scatter3d", mode="markers", color=temp)

data("airmiles")
plot_ly(x=time(airmiles), y=airmiles)

library(tidyr)
library(dplyr)
data("EuStockMarkets")
head(EuStockMarkets)

stocks<-as.data.frame(EuStockMarkets) %>% gather(index, price) %>% mutate(time = rep(time(EuStockMarkets), 4))

plot_ly(stocks, x=stocks$time, y=stocks$price, color = stocks$index)