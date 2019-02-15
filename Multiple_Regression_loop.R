# Multicore processing
install.packages("doMC")
library(doMC)
registerDoMC(cores=4)

# Install Packages
install.packages("pacman")
library("pacman")
pacman::p_load("e1071", "lattice", "ggplot2", "caret", "corrplot", "gbm", "dplyr","e1071", "plotly")

# Import Data
setwd("/Users/denizminican/Dropbox/03-Data_and_Coding/Ubiqum/DAII_3/")
getwd()
Existing <- read.csv("existingproductattributes2017.csv")
summary(Existing)
duplicated(Existing)

##### Playing with data ####
#Predictions on important variables
data2 <- filter(
  Existing, ProductType %in% c(
    "PC","Laptop","Smartphone","Netbook"
    )
  )
data2

####Data Cleaning####
# Remove Warranty
data3 <- Existing[!duplicated(Existing[,c("ProductType","PositiveServiceReview", 
                                  "ProductDepth", "ShippingWeight", 
                                  "Volume", "x4StarReviews"
                                  )
                               ]
                          ),
              ]
data3

# Remove outliers
outliers <- boxplot(Existing$Volume)$out
data3[which(Existing$Volume %in% outliers),]
data3 <- data3[-which(data3$Volume %in% outliers),]
data3

# Remove 5starReviews and BestSellerRank
drops <- c("x5StarReviews","BestSellersRank")
data3 <- data3[, !(names(data3) %in% drops)]
data3

####Dummies and Correlation####
#Create Dummy Variables
#What's happening here?
newDataF <- dummyVars(" ~ .", data = data3)
data4 <- data.frame(predict(newDataF, newdata = data3))
data4
#Correlation Matrix
corrData <- cor(data4)
corrData
corrplot(corrData, type = "upper", tl.pos = "td",
         method = "circle", tl.cex = 0.5, tl.col = 'black',
         order = "hclust", diag = FALSE)

####MODELING####
#p=0.75 gives variable 6: ProductType.Netbook has no variation.
set.seed(123)
inTraining <- createDataPartition(
  data4$Volume,
  p = 0.80,
  list = FALSE
)
trainSet <- data4[inTraining,]
testSet <- data4[-inTraining,]


fitControl <- trainControl(
  method = "repeatedcv",
  predictionBounds = c(0,NA),
  number = 10,
  repeats = 2
)

#Importance of 4StarReviews and PositiveServiceReview
ggplot(
  data2, aes(
    x=ProductType, 
    y=Volume, 
    size=x4StarReviews, 
    color=PositiveServiceReview
  )
)+
  geom_point()+
  ylim(0,1500)

####MODEL LOOPS####
#List of models 
models = c("lm","gbm","rf","svmLinear2","knn")
trainedModels = list()
#For loop (TO INCLUDE LATER)
for(i in models){
  set.seed(123)
  trainedModel <- train(Volume ~ x4StarReviews+PositiveServiceReview, 
        data=trainSet, 
        method= i, 
        metric="RMSE", 
        trControl=fitControl
        )
  trainedModels[[i]] <- trainedModel
}
trainedModels
#Summary of Models
results <- resamples(trainedModels)
results
summary(results)
#Performance Metric Comparisons
bwplot(results,
       scales = list(relation = 'free'),
       xlim = list(c(0, 600), c(0, 700), c(0,1)),
       layout = c(1,3),
       fill=c("yellow")
)

#How to include this in loop?
####TestSet Predictions####
#LINEAR MODEL 
Volume.predLM <-predict(trainedModels$lm, testSet)
Volume.predLM <- as.integer(Volume.predLM)
Volume.predLM

DT1 <- data.frame(testSet$ProductNum, testSet$Volume)
DT2 <- data.frame(Volume.predLM)
DT_LM <- cbind(DT1, DT2)
DT_LM
DT1

# #GBM manualgrid
# gbmGrid <-  expand.grid(interaction.depth = (1:5)*2, 
#                         n.trees = (1:25)*50, 
#                         shrinkage = 0.1,
#                         n.minobsinnode=10
#                         )
# set.seed(123)
# gbmFit2 <- train(Volume ~x4StarReviews+PositiveServiceReview,
#                  data = trainSet, 
#                  method = "gbm", 
#                  trControl = fitControl, 
#                  verbose = FALSE, 
#                  tuneGrid = gbmGrid
#                  )

#SupportVectorMachine
Volume.predSVM <-predict(trainedModels$svmLinear2, testSet)
Volume.predSVM <- as.integer(Volume.predSVM)
Volume.predSVM

DT3 <- data.frame(Volume.predSVM)
DT_SVM <- cbind(DT1, DT3)
DT_SVM
#GradientBoostedMachine 
Volume.predGBM<-predict(trainedModels$gbm, testSet)
Volume.predGBM <- as.integer(Volume.predGBM)
Volume.predGBM

DT4 <- data.frame(Volume.predGBM)
DT_GBM <- cbind(DT1, DT4)
DT_GBM
#Our model suffers from heteroscedasticity
ggplot(DT_GBM, aes(x=Volume.predGBM, 
                   y=testSet.Volume
                   )
       )+
  geom_point()+
  geom_abline(intercept=0, slope=1, color="blue")+
  ggtitle("GBM Volume Predictions vs Real Volumes")+
  ylab("Volume")+xlab("GBM Volume Predictions")
  # geom_smooth(method = "lm", se = FALSE)

#RandomForest
Volume.predRF <-predict(trainedModels$rf, testSet)
Volume.predRF <- as.integer(Volume.predRF)
Volume.predRF

DT5 <- data.frame(Volume.predRF)
DT_RF <- cbind(DT1, DT5)
DT_RF

#kNN
Volume.predKNN <-predict(trainedModels$knn, testSet)
Volume.predKNN <- as.integer(Volume.predkNN)
Volume.predKNN

DT6 <- data.frame(Volume.predKNN)
DT_KNN <- cbind(DT1, DT6)
DT_KNN

##Combining Models
#DO A LOOP
DT_GBM <- dplyr::rename(DT_GBM, Volume = Volume.predGBM)
DT_GBM$model <- "GBM"

DT_SVM <- dplyr::rename(DT_SVM, Volume = Volume.predSVM)
DT_SVM$model <- "SVM"

DT_LM <- dplyr::rename(DT_LM, Volume = Volume.predLM)
DT_LM$model <- "LM"

DT_RF<- dplyr::rename(DT_RF, Volume = Volume.predRF)
DT_RF$model <- "RF"

DT_KNN <- dplyr::rename(DT_KNN, Volume = Volume.predKNN)
DT_KNN$model <- "KNN"
rename
DT_all <-  rbind(DT_GBM,DT_SVM,DT_LM, DT_RF, DT_KNN)
DT_all

##Result Analysis (From TestSet)
#Plotting volume predictions together
ggplot(DT_all, aes(x=testSet.Volume, y=Volume, color=model))+
  geom_point()+
  geom_abline(slope=1, intercept= 0, color="purple")+
  xlab("Real Volume")+
  ylab("Volume Predictions")+
  ggtitle("Prediction Comparison Plot")

#
postResample(Volume.predGBM, testSet$Volume)
postResample(Volume.predKNN, testSet$Volume)
postResample(Volume.predRF, testSet$Volume)
####PREDICTIONS####
NewProducts <- read.csv("newproductattributes2017.csv")
#Drop BestSellerRank and 5StarReviews
NewProducts <- NewProducts[, !(names(NewProducts) %in% drops)]
NewProducts

#GBM
VolumeNew.GBM<-predict(trainedModels$gbm, NewProducts)
VolumeNew.GBM <- as.integer(VolumeNew.GBM)
VolumeNew.GBM
Volume_pred <- data.frame(VolumeNew.GBM)
NewProductsFilled <- cbind(NewProducts,Volume_pred)
NewProductsFilled <- filter(
  NewProductsFilled, ProductType %in% c(
    "PC","Laptop","Smartphone","Netbook"
  )
)

#Adding Product Volume Percentages per Type
NewProductsFilled <- NewProductsFilled[,c(1,2,3,17)]
NewProductsFilled

NewProductsFilled <- NewProductsFilled %>%
  group_by(ProductType) %>%
  mutate(VolumePercent = sum(VolumeNew.GBM))

NewProductsFilled <- NewProductsFilled %>%
  mutate(VolumeP = (VolumeNew.GBM/VolumePercent))

NewProductsFilled

#Sales Predictions per Product Type
NewProductsFilled$ProductNum <- reorder(NewProductsFilled$ProductNum, 
                                      NewProductsFilled$VolumeNew.GBM
                                      )
NewProductsFilled$ProductNum <- factor(NewProductsFilled$ProductNum, 
                                     levels=rev(levels(NewProductsFilled$ProductNum
                                                       )
                                                )
                                     )
ggplot(NewProductsFilled, aes(x= ProductType,
                              y= VolumeNew.GBM,
                              fill=as.factor(ProductNum)
                              )
       )+
  geom_col(position = "stack")+
  labs(x= "Product Type", y="Volume Predictions", fill="Product Number")+
  ggtitle("Predicted Volumes per Product")

#Total Profit Graph
NewProducts$ProductNum <- as.factor(NewProducts$ProductNum)
NP_to_join <- NewProducts[,c(1,2,15)]
NP_w_PT <- left_join(NewProductsFilled, NP_to_join, by="ProductNum")
NP_w_PT$TotalProfit <- NP_w_PT$Price * NP_w_PT$ProfitMargin * NP_w_PT$Price
NP_w_PT

NP_w_PT$ProductNum <- reorder(NP_w_PT$ProductNum, 
                              NP_w_PT$TotalProfit
                              )
NP_w_PT$ProductNum <- factor(NP_w_PT$ProductNum, 
                                       levels=rev(levels(NP_w_PT$ProductNum
                                                         )
                                                  )
                             )
ggplot(NP_w_PT, aes(x= ProductType.x,
                              y= TotalProfit,
                              fill=as.factor(ProductNum)
                    )
       )+
  geom_col(position = "stack")+
  labs(x= "Product Type", y="Total Profit", fill="Product Number")+
  ggtitle("Total Profit per Category")


#Pie Charts - Products per Category 

ggplot(NewProductsFilled, aes(x= "",
                    y= VolumeP,
                    fill=as.factor(ProductNum)
                    )
       )+
   geom_bar(width = 1, stat = "identity")+
   coord_polar("y", start=0)+
   labs(x= "", y="Total Volume", fill="Product Number")+
   ggtitle("Product Importance per Category")+
   facet_grid("ProductType")

ggplot(NewProductsFilled, aes(x= "",
                              y= VolumeP,
                              fill=as.factor(ProductNum)
)
)+
  geom_bar(width = 1, stat = "identity")+
  coord_polar("y", start=0)+
  labs(x= "Product Number", y="Total Profit", fill="Product Number")+
  ggtitle("Total Profit per Category")+
  facet_grid("ProductType")

install.packages("waffle")
install.packages("extrafontdb")
install.packages("Rttf2pt1")


