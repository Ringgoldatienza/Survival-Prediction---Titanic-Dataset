#Predicting Titanic Surviror Project
#by Ringgold P. Atienza

################################################################################
#Install packages and load dataset

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

#Load titanic training dataset
Titanic_Train <- read.csv("C://Users//ADMIN//Documents//GitHub//Titanic//Dataset//train.csv")

#Load titanic test dataset
Titanic_Test <- read.csv("C://Users//ADMIN//Documents//GitHub//Titanic//Dataset//test.csv")

################################################################################
#Inspect dataset

#Show column names and class for each column
str(Titanic_Train)
colnames(Titanic_Train)
head(Titanic_Train)

#Plot survive variable
ggplot(Titanic_Train, aes(Survived)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous(breaks = seq(0, 2, 1))

#Plot pasenger class variables
ggplot(Titanic_Train, aes(Pclass)) +
         geom_histogram(binwidth = 1, color = "black")

#Plot sex variable
ggplot(Titanic_Train, aes(Sex)) +
  geom_histogram(binwidth = 1, 
                 stat = "Count",
                 color = "black")

#Plot age Variable
#We found 177 NA values in the age variable
ggplot(Titanic_Train, aes(Age)) +
  geom_histogram(binwidth = 1, color = "black")

#Plot siblings variable
ggplot(Titanic_Train, aes(SibSp)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous(breaks = seq(0, 8, 1))

#Plot parch variable
ggplot(Titanic_Train, aes(Parch)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous(breaks = seq(0, 6, 1))

#Count number of distinct tickets
n_distinct(Titanic_Train$Ticket)

##Plot fare variable
ggplot(Titanic_Train, aes(Fare)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_y_log10()

#Count number of distince Cabin
n_distinct(Titanic_Train$Cabin)

#Plot embarkation place variable
#Found 2 observation without embarkation
ggplot(Titanic_Train, aes(Embarked)) +
  geom_histogram(stat = "Count",
                 color = "black")

################################################################################
#Mutate NAs in the dataset

#Count NAs in the data frame
na_count <- sapply(Titanic_Train, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count

#Transform embarked data (NAs into S)
Titanic_Train$Embarked[Titanic_Train$Embarked == ""] <- "S"

#Inspect correlation of Age to other variables

#Age-Survived plot
Titanic_Train %>% ggplot(aes(as.factor(Survived), Age)) + 
  geom_boxplot()

#Age-Pclass plot
Titanic_Train %>% ggplot(aes(Pclass, Age)) + 
  geom_point(alpha = 0.5) +
  scale_x_continuous(breaks = seq(1, 3, 1), labels = seq(1, 3, 1))

#Age-Sex plot
Titanic_Train %>% ggplot(aes(Sex, Age)) + 
  geom_point(alpha = 0.5)

#Age-SibSp plot
Titanic_Train %>% ggplot(aes(SibSp, Age)) + 
  geom_point(alpha = 0.5) +
  scale_x_continuous(breaks = seq(1, 8, 1), labels = seq(1, 8, 1))

#Age-Parch plot
Titanic_Train %>% ggplot(aes(Parch, Age)) + 
  geom_point(alpha = 0.5) +
  scale_x_continuous(breaks = seq(1, 6, 1), labels = seq(1, 6, 1))

#Age-Fare plot
Titanic_Train %>% ggplot(aes(Fare, Age)) + 
  geom_point(aes(colour = Survived,
                 size = Parch)) +
  facet_wrap(~ Sex, nrow = 2)
  
#Age-Embarked plot
Titanic_Train %>% ggplot(aes(Embarked, Age)) + 
  geom_point(alpha = 0.5)


#Set Subset Data for Age
Age_Subset <- Titanic_Train[c('Survived','Pclass','Age', 'Sex',
                              'SibSp', 'Parch', 'Fare', 'Embarked')]

#Count NAs in the data frame
na_count <- sapply(Age_Subset, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count

################################################################################
#Least square estimate

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

mu <- mean(Age_Subset$Age, na.rm = TRUE)

rmse_baseline <- RMSE(Age_Subset$Age, mu)

rmse_baseline_step  <- data.frame(Variable = "Baseline (mu)", 
                                  RMSE = rmse_baseline, 
                                  Difference = rmse_baseline - rmse_baseline)

#######################################
#Predict mu + Fare_avgs

Fare_avgs <- Age_Subset %>% 
  group_by(Fare) %>% 
  summarize(b_f = mean(Age - mu, na.rm = TRUE))

#Count NAs in the Fare_avgs
na_count <- sapply(Fare_avgs, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
sum(na_count)

#Transform NAs in Fare_avgs using rnorm
mu_Fare_avgs <- mean(Fare_avgs$b_f, na.rm = TRUE)
stdev_Fare_avgs <- sd(Fare_avgs$b_f, na.rm = TRUE)

#Create sequence of random number based on normal distibution
pred_seq <- rnorm(sum(na_count), mean = mu_Fare_avgs, sd = (stdev_Fare_avgs/2))

#Insert random number in NA values
Fare_avgs$b_f[is.na(Fare_avgs$b_f)] <- pred_seq

#######################################
#Predict mu + Fare + Parch

Parch_avgs <- Age_Subset %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  group_by(Parch) %>% 
  summarize(b_a = mean(Age - mu - b_f, na.rm = TRUE))

#######################################
#Predict mu + Fare + Parch + Survived

Survived_avgs <- Age_Subset %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  group_by(Survived) %>% 
  summarize(b_s = mean(Age - mu - b_f - b_a, na.rm = TRUE))

#######################################
#Predict mu + Fare + Parch + Survived + SibSp

SibSp_avgs <- Age_Subset %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  left_join(Survived_avgs, by = "Survived") %>%
  group_by(SibSp) %>% 
  summarize(b_i = mean(Age - mu - b_f - b_a - b_s, na.rm = TRUE))

#Count NAs in the SibSp_avgs
na_count <- sapply(SibSp_avgs, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
sum(na_count)

#Plot predicted age vs actual age
ggplot() +
  geom_point(data = SibSp_avgs, aes(SibSp, b_i))
  
#Linear trend model
SibSp_lt <- lm(b_i ~ SibSp, SibSp_avgs)
SibSp <- c(0:8)
SibSp  <- data.frame(SibSp)
sibSp_pred <- predict(SibSp_lt, SibSp)
sibSp_pred <- data.frame(sibSp_pred)

#Insert average number in NA values
SibSp_avgs$b_i[is.na(SibSp_avgs$b_i)] <- sibSp_pred[8,]

#Plot predicted age vs actual age
ggplot() +
  geom_point(data = SibSp_avgs, aes(SibSp, b_i)) +
  geom_line(data = sibSp_pred, aes(c(0:8), sibSp_pred), color = "Red")

#######################################
#Predict mu + Fare + Parch + Survived + SibSp + Pclass

Pclass_avgs <- Age_Subset %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  left_join(Survived_avgs, by = "Survived") %>%
  left_join(SibSp_avgs, by = "SibSp") %>%
  group_by(Pclass) %>% 
  summarize(b_p = mean(Age - mu - b_f - b_a - b_s - b_i, na.rm = TRUE))

#######################################
#Predict mu + Fare + Parch + Survived + SibSp + Pclass + Sex

Sex_avgs <- Age_Subset %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  left_join(Survived_avgs, by = "Survived") %>%
  left_join(SibSp_avgs, by = "SibSp") %>%
  left_join(Pclass_avgs, by = "Pclass") %>%
  group_by(Sex) %>% 
  summarize(b_e = mean(Age - mu - b_f - b_a - b_s - b_i - b_p, na.rm = TRUE))

predicted_age <- Titanic_Train %>%
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  left_join(Survived_avgs, by = "Survived") %>%
  left_join(SibSp_avgs, by = "SibSp") %>%
  left_join(Pclass_avgs, by = "Pclass") %>%
  left_join(Sex_avgs, by = "Sex") %>%
  mutate(pred = mu + b_f + b_a + b_s + b_i + b_p + b_e) %>%
  pull(pred)

#Put predicted age into the dataset
Titanic_Train_Clean <- cbind(Titanic_Train, predicted_age)

#Input predicted values to NAs in the Original Dataset
Titanic_Train_Clean<- Titanic_Train_Clean %>%
  mutate(Age = ifelse(is.na(Age), predicted_age, Age))

#Round Age Values
Titanic_Train_Clean <- Titanic_Train_Clean %>% 
  mutate(Age = if_else(Age < 1, Age, round(Age, 0)))

#Plot predicted age vs actual age
ggplot(Titanic_Train_Clean, aes(Age, predicted_age)) +
  geom_point() +
  geom_abline(intercept = TRUE)

#Plot age Variable
ggplot(Titanic_Train_Clean, aes(Age)) +
  geom_histogram(binwidth = 1, color = "black")

################################################################################
#Inspect survival variable

#Age-Survived plot
Titanic_Train_Clean %>% ggplot(aes(as.factor(Survived), Age)) + 
  geom_boxplot() +
  labs(x = "Survived", y = "Age")
  
#Pclass-Survived plot
Titanic_Train_Clean %>% ggplot(aes(as.factor(Survived))) + 
  geom_bar(aes(fill = as.factor(Pclass))) +
  labs(x = "Survived", fill = "Pclass")

#Sex-Survived plot
Titanic_Train_Clean %>% ggplot(aes(as.factor(Survived))) +
  geom_bar(aes(fill = Sex)) +
  labs(x = "Survived", fill = "Sex")

#SibSp-Survived plot
Titanic_Train_Clean %>% ggplot(aes(y = as.factor(Survived))) +
  geom_bar(aes(fill = as.factor(SibSp))) +
    labs(y = "Survived", fill = "SibSp")

#Parch-Survived plot
Titanic_Train_Clean %>% ggplot(aes(y = as.factor(Survived))) +
  geom_bar(aes(fill = as.factor(Parch))) +
  labs(y = "Survived", fill = "Parch")

#Fare-Survived plot
Titanic_Train_Clean %>% ggplot(aes(as.factor(Survived), Fare)) + 
  geom_boxplot() +
  labs(x = "Survived", y = "Fare")

#Embarked-Survived plot
Titanic_Train_Clean %>% ggplot(aes(as.factor(Survived))) + 
  geom_bar(aes(fill = as.factor(Embarked))) +
  labs(x = "Survived", fill = "Embarked")

################################################################################
#Mutate Test-Set missing values

#Count NAs
na_count <- sapply(Titanic_Test, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count

#Mutate NAs of Fare variable
Titanic_Test$Fare[is.na(Titanic_Test$Fare)] <- median(Titanic_Test$Fare, na.rm = TRUE)

################################################################################
#Mutate Age Values
#Least square estimate

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

mu <- mean(Titanic_Test$Age, na.rm = TRUE)

#######################################
#Predict mu + Fare_avgs

Fare_avgs <- Titanic_Test %>% 
  group_by(Fare) %>% 
  summarize(b_f = mean(Age - mu, na.rm = TRUE))

#Count NAs in the Fare_avgs
na_count <- sapply(Fare_avgs, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
sum(na_count)

#Transform NAs in Fare_avgs using rnorm
mu_Fare_avgs <- mean(Fare_avgs$b_f, na.rm = TRUE)
stdev_Fare_avgs <- sd(Fare_avgs$b_f, na.rm = TRUE)

#Create sequence of random number based on normal distibution
pred_seq <- rnorm(sum(na_count), mean = mu_Fare_avgs, sd = (stdev_Fare_avgs/2))

#Insert random number in NA values
Fare_avgs$b_f[is.na(Fare_avgs$b_f)] <- pred_seq

#######################################
#Predict mu + Fare + Parch

Parch_avgs <- Titanic_Test %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  group_by(Parch) %>% 
  summarize(b_a = mean(Age - mu - b_f, na.rm = TRUE))

#Count NAs in the Parch_avgs
na_count <- sapply(Parch_avgs, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
sum(na_count)

#Plot predicted age vs actual age
ggplot() +
  geom_point(data = Parch_avgs, aes(Parch, b_a))

#Linear trend model
Parch_lt <- lm(b_a ~ Parch, Parch_avgs)
Parch <- c(0:8)
Parch  <- data.frame(Parch)
Parch_pred <- predict(Parch_lt, Parch)
Parch_pred <- data.frame(Parch_pred)

#Insert average number in NA values
Parch_avgs$b_a[is.na(Parch_avgs$b_a)] <- Parch_pred[8,]

#######################################
#Predict mu + Fare + Parch + SibSp

SibSp_avgs <- Titanic_Test %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  group_by(SibSp) %>% 
  summarize(b_i = mean(Age - mu - b_f - b_a, na.rm = TRUE))

#######################################
#Predict mu + Fare + Parch + Survived + SibSp + Pclass

Pclass_avgs <- Titanic_Test %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  left_join(SibSp_avgs, by = "SibSp") %>%
  group_by(Pclass) %>% 
  summarize(b_p = mean(Age - mu - b_f - b_a - b_i, na.rm = TRUE))

#######################################
#Predict mu + Fare + Parch + Survived + SibSp + Pclass + Sex

Sex_avgs <- Titanic_Test %>% 
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  left_join(SibSp_avgs, by = "SibSp") %>%
  left_join(Pclass_avgs, by = "Pclass") %>%
  group_by(Sex) %>% 
  summarize(b_e = mean(Age - mu - b_f - b_a - b_i - b_p, na.rm = TRUE))

predicted_age <- Titanic_Test %>%
  left_join(Fare_avgs, by = "Fare") %>%
  left_join(Parch_avgs, by = "Parch") %>%
  left_join(SibSp_avgs, by = "SibSp") %>%
  left_join(Pclass_avgs, by = "Pclass") %>%
  left_join(Sex_avgs, by = "Sex") %>%
  mutate(pred = mu + b_f + b_a + b_i + b_p + b_e) %>%
  pull(pred)

#Put predicted age into the dataset
Titanic_Test_Clean <- cbind(Titanic_Test, predicted_age)

#Input predicted values to NAs in the Original Dataset
Titanic_Test_Clean<- Titanic_Test_Clean %>%
  mutate(Age = ifelse(is.na(Age), predicted_age, Age))

#Round Age Values
Titanic_Test_Clean <- Titanic_Test_Clean %>% 
  mutate(Age = if_else(Age < 1, Age, round(Age, 0)))

#Plot predicted age vs actual age
ggplot(Titanic_Test_Clean, aes(Age, predicted_age)) +
  geom_point() +
  geom_abline(intercept = TRUE)

#Plot age Variable
ggplot(Titanic_Test_Clean, aes(Age)) +
  geom_histogram(binwidth = 1, color = "black")

################################################################################
#Random Forest

#Assign data into train (80%) and test (20%) sets.
set.seed(2022, sample.kind = 'Rounding')
test_index <- createDataPartition(y = Titanic_Train_Clean$Survived, times = 1, 
                                  p = 0.2, list = FALSE)
Titanic_Train_Clean_R <- Titanic_Train_Clean[-test_index,]
Titanic_Test_Clean_R <- Titanic_Train_Clean[test_index,]

Titanic_Train_Subset <- Titanic_Train_Clean_R[c('Survived','Pclass','Age',
                                                'Sex','SibSp', 'Parch', 'Fare', 
                                                'Embarked')]

Titanic_Test_Subset <- Titanic_Test_Clean_R[c('Survived','Pclass','Age',
                                              'Sex','SibSp', 'Parch', 'Fare', 
                                              'Embarked')]

Titanic_Train_Subset <- Titanic_Train_Subset %>%
  mutate(Survived = as.factor(Survived))

Titanic_Test_Subset <- Titanic_Test_Subset %>%
  mutate(Survived = as.factor(Survived))

#Run randomForest
rf <- randomForest(Survived~., data = Titanic_Train_Subset, proximity = TRUE)
print(rf)

#Create prediction values on validation-set
predicted_survived <- predict(rf, Titanic_Test_Subset)

#Create confusion Matrix
confusionMatrix(predicted_survived, Titanic_Test_Subset$Survived)


################################################################################
#Final

#Run Final Testing
Titanic_Train_Clean <- Titanic_Train_Clean[c('Survived','Pclass','Age',
                                                'Sex','SibSp', 'Parch', 'Fare', 
                                                'Embarked')]

Titanic_Train_Clean <- Titanic_Train_Clean %>%
  mutate(Survived = as.factor(Survived))

Titanic_Test_Clean <- Titanic_Test_Clean %>%
  mutate(Survived = as.factor(Survived))

#Run randomForest
rf <- randomForest(Survived~., data = Titanic_Train_Clean, proximity = TRUE)
print(rf)

#Create prediction values on validation-set
Survived <- predict(rf, Titanic_Test_Clean)

#Bind Prediction and Test set
Prediction_Final <- cbind(Titanic_Test_Clean, Survived)

#Finalize data
Prediction_Final <- Prediction_Final[c('PassengerId','Survived')]

#Save final into CSV
write.csv(Prediction_Final,"C:\\Users\\ADMIN\\Documents\\GitHub\\Titanic\\submission.csv", 
          row.names = TRUE)

