library(caret)
library(pROC)
library(tidyverse)
library(MASS)
library(car)

log_data <- read.csv('no_dummies.csv')
full_data <- read.csv('full_data.csv')
View(full_data)
View(log_data)
attach(log_data)

str(log_data)

log_data$completed <- as.factor(completed)
log_data$social <- as.factor(social)
log_data$mobile <- as.factor(mobile)
log_data$web <- as.factor(web)
log_data$email <- as.factor(email)

# Baseline accuracy 
(sum(full_data$completed))/(nrow(full_data))
# 0.6328766

##### All Ages #####
nrow(log_data) * 0.2
train <- log_data[1:20500,]
test <- log_data[20501:102693,]


log.model3 <- glm(completed ~ reward_y + difficulty + duration + social + mobile + web + age +
                    income + offer_type, family = binomial,
                  data = train)
summary(log.model3)
vif(log.model3)

log.model4 <- glm(completed ~ difficulty + duration + social + mobile + web + age +
                    income + offer_type, family = binomial,
                  data = train)
summary(log.model4)


log_prob2 = predict(log.model3, test, type = 'response')
log_predict2= ifelse(log_prob2 >= 0.5, '1', '0')
table(log_predict2, test$completed)

levels(test$completed)
levels(log_predict2)
log_predict2 = as.factor(log_predict2)

confusionMatrix(log_predict2, test$completed)
#accuracy - 0.8002

test_roc = roc(test$completed ~ log_prob2, plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc)


##### College Age Group #####
college_age <- subset(log_data, age <= 35)
View(college_age)


## Baseline for college aged
complete.int <- as.integer(college_age$completed) - 1
complete.int
(sum(complete.int))/(nrow(college_age))
# 0.5298112

nrow(college_age) * 0.2
college_train <- college_age[1:3100, ]
college_test <- college_age[3101: nrow(college_age), ]

college.log.model <- glm(completed ~ reward_y + difficulty + duration + social + mobile + web + 
                           age + income + offer_type, family = binomial,
                         data = college_train)
summary(college.log.model)
vif(college.log.model)

college.log.model2 <- glm(completed ~  difficulty + duration + social + mobile
                         + income + offer_type, family = binomial,
                         data = college_train)
summary(college.log.model2)

log_prob_college = predict(college.log.model, college_test, type = 'response')
log_predict_college= ifelse(log_prob_college >= 0.5, '1', '0')
table(log_predict_college, college_test$completed)

levels(college_test$completed)
levels(log_predict_college)
log_predict_college = as.factor(log_predict_college)

confusionMatrix(log_predict_college, college_test$completed)
#accuracy - 0.6953

test_roc2 = roc(college_test$completed ~ log_prob_college, plot = TRUE, print.auc = TRUE)
as.numeric(test_roc2$auc)


##### Linear Discriminant Analysis #####

#Fit the model
lda.model <- lda(completed ~ reward_y + difficulty + duration + social + mobile + web + age + income + 
                   offer_type, data = train)
lda.model

# Prior Probabilities - 0.6417073 (baseline)

# Make predictions
pred <- lda.model %>% predict(test)
# Model accuracy - 0.7911136
mean(pred$class == test$completed)

lda.data <- cbind(train, predict(lda.model)$x)
lda.data
ggplot(lda.data, aes(LD1)) +
  geom_histogram(aes(fill = completed))


lda.model.col <- lda(completed ~ reward_y + difficulty + duration + social + mobile + web + age + income + 
                   offer_type, data = college_train)
lda.model.col

# Priori Probabilities - 0.556129

pred.col <- lda.model.col %>% predict(college_test)
# Model accuracy - 0.6948399
mean(pred.col$class == college_test$completed)

lda.data.col <- cbind(college_train, predict(lda.model.col)$x)
lda.data.col
ggplot(lda.data.col, aes(LD1)) +
  geom_histogram(aes(fill = completed))
