# Attempt 0.0
#Model: glmnet
#Oversampling: no
#Fourier: no
#Features: statistic of the boxtest applied to the autocorrelation series

library(data.table)
library(glmnet)
library(progress)
library(caret)

data = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")

num_ts <- 24

new_data <- matrix(NA, nrow(data), 24)

total_steps <- 24*nrow(data)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data)){
  for (j in 1:24){
    ts <- as.numeric(data[i, (5+(2048*(j-1))):(2053+(2048*(j-1)))])
    ljung_box_test <- Box.test(ts, lag = 10, type = "Ljung-Box")$statistic
    new_data[i, j] <- as.numeric(ljung_box_test)
    pb$tick()
  }
}

new_data <- data.frame(new_data)
new_data$y <- data$y
head(new_data)

trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]

model_1 <- glmnet(as.matrix(train[, 1:(ncol(train)-1)]), train$y, lambda=0.0002, alpha=0, family="multinomial")
predictions_1 <- predict(model_1, as.matrix(validation[, 1:(ncol(validation)-1)]), type="class")

table(validation$y, predictions_1)

#Attempt 0.1¶
#Model: glmnet
#Oversampling: no
#Fourier: no
#Features:
#sd, median, iqr, diff(range))
library(data.table)
library(glmnet)
library(progress)
library(caret)

num_ts <- 24 # number of time series

new_data <- matrix(NA, nrow(data), num_ts*4) # new dataset based on the features

total_steps <- num_ts*nrow(data) 
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data)){
  for (j in 1:num_ts){
    ts <- as.numeric(data[i, (5+(2048*(j-1))):(2053+(2048*(j-1)))])
    new_data[i, j] <- median(ts)
    new_data[i, (j+num_ts)] <- sd(ts)
    new_data[i, (j+num_ts*2)] <- IQR(ts)
    new_data[i, (j+num_ts*3)] <- diff(range(ts))
    pb$tick()
  }
}
new_data <- data.frame(new_data)
new_data$y <- data$y
head(new_data)

trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]

model <- glmnet(as.matrix(train[, 1:(ncol(train)-1)]), train$y, lambda=0.003, alpha=0.5, family="multinomial")
predictions <- predict(model, as.matrix(validation[, 1:(ncol(validation)-1)]), type="class")

table(validation$y, predictions)

# Attempt 1
# Model: glmnet
# 
# Oversampling: yes
# 
# Fourier: until 50 and then removed elements with abs(x) < 0.01
# 
# Features: Fourier series coefficients

library(data.table)
library(glmnet)
library(smotefamily)
gino = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
mat = as.matrix(gino[,6:49157])
mat_fou <- matrix(NA,663,(101)*24)

N_data <- length(mat[1,1:2048])
N <- 50

for(i in 1:663){
  
  for(j in 1:24){
    data <- mat[i,((j-1)*2048+1):(j*2048)]
    
    a0 <- sum(data)/N_data
    
    an <- numeric(N)
    bn <- numeric(N)
    
    for (k in 1:N) {
      an[k] <- 2 * sum(data * cos(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
      bn[k] <- 2 * sum(data * sin(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
    }
    mat_fou[i,((j-1)*101+1):(j*101)] <- c(a0,an,bn)
  }
}

dat_fou <- data.frame(mat_fou)
dat_fou$y <- gino$y

train_smote1 <- SMOTE(X = subset(dat_fou, y %in% c(1, 2)),
                      target = subset(dat_fou$y, dat_fou$y %in% c(1, 2)),
                      K = 3, dup_size = 0)

table(train_smote1$data$y)
train_smote2 <- SMOTE(X = subset(dat_fou, y %in% c(1, 3)),
                      target = subset(dat_fou$y, dat_fou$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

table(train_smote2$data$y)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data[train_smote2$data$y==3,])[,-2426]
table(train_smote_combined$y)

sam <- sample(dim(train_smote_combined)[1],1400, replace=F)

pino <- cv.glmnet(as.matrix(train_smote_combined[sam,-2425]), train_smote_combined[sam,2425], family="multinomial")

pino <- glmnet(train_smote_combined[,-2425], train_smote_combined[,2425],
               family="multinomial", alpha=1, lambda=0.003705)


test <- fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")

mat_test = as.matrix(test[,5:49156])
mat_fou_test <- matrix(NA,355,(101)*24)

N_data <- length(mat[1,1:2048])
N <- 50

for(i in 1:355){
  print(i)
  for(j in 1:24){
    data <- mat_test[i,((j-1)*2048+1):(j*2048)]
    
    a0 <- sum(data)/N_data
    
    an <- numeric(N)
    bn <- numeric(N)
    
    for (k in 1:N) {
      an[k] <- 2 * sum(data * cos(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
      bn[k] <- 2 * sum(data * sin(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
      
    }
    
    mat_fou_test[i,((j-1)*101+1):(j*101)] <- c(a0,an,bn)
  }
}

predictions <- predict(pino, mat_fou_test, type="class")
submission <- data.frame(id=test$id, y=as.numeric(predictions))

write.csv(submission, file = "sub1.csv", row.names = FALSE)

# Attempt 2
# Model: glmnet
# 
# Oversampling: yes
# 
# Fourier: yes with fft
# 
# Features:
#   
#   Energy
# 
# Entropy
# 
# Centroid frequency
# 
# Peak frequency


library(data.table)
library(glmnet)
library(smotefamily)
library(progress)
library(caret)


# Oversampling with SMOTE
subset_df <- data[, 6:ncol(data)]

smote1 <- data.frame(y = data$y, subset_df)

train_smote1 <- SMOTE(X = subset(smote1, y %in% c(1, 2)),
                      target = subset(smote1$y, smote1$y %in% c(1, 2)),
                      K = 3, dup_size = 0)

table(train_smote1$data$y)

train_smote2 <- SMOTE(X = subset(smote1, y %in% c(1, 3)),
                      target = subset(smote1$y, smote1$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

table(train_smote2$data$y)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data)

data_plus <- train_smote_combined[!duplicated(train_smote_combined), ]

# Fourier & Features extraction for train set
num_ts <- 24

new_data <- matrix(NA, nrow(data_plus), num_ts*4)

total_steps <- num_ts*nrow(data_plus)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data_plus)){
  for (j in 1:num_ts){
    ts <- as.numeric(data_plus[i, (1+(2048*(j-1))):(2049+(2048*(j-1)))])
    fourier_series <- fft(ts)
    new_data[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_data[i, (j+num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_data[i, (j+num_ts*2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_data[i, (j+num_ts*3)] <- which.max(abs(fourier_series)) # peak frequency
    
    pb$tick()
  }
}

new_data <- data.frame(new_data)
new_data$y <- data_plus$y
head(new_data)

# Partitioning the dataset

set.seed(123)

trainIndex <- createDataPartition(new_data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]
# model 
model <- glmnet(as.matrix(train[, 1:(ncol(train)-1)]), train$y, lambda=0.001,
                maxit = 1e+09, alpha=1, family="multinomial")

predictions <- predict(model, as.matrix(validation[, 1:(ncol(validation)-1)]), type="class")

table(validation$y, predictions)

# Fourier & Features extraction for test set
test <- fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")
new_test <- matrix(NA, nrow(test), num_ts*4)

total_steps <- num_ts*nrow(new_test)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(test)){
  for (j in 1:num_ts){
    ts <- as.numeric(test[i, (5+(2048*(j-1))):(2052+(2048*(j-1)))])
    fourier_series <- fft(ts)
    new_test[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_test[i, (j+num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_test[i, (j+num_ts*2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_test[i, (j+num_ts*3)] <- which.max(abs(fourier_series)) # peak frequency
    pb$tick()
  }
}

# Predictions on test set
predictions <- predict(model, new_test, type="class")
table(predictions)
submission <- data.frame(id=test$id, y=as.numeric(predictions))
write.csv(submission, file = "sub3.csv", row.names = FALSE)


# Now we have features, will it be enough? Not yet.
# 
# Also in this case we tried different models and different parameters, one surprised us (in the following subs...)
# 


# Attempt 2.1
# Model: Gradient Boosting
# 
# Oversampling: yes
# 
# Fourier: yes
# 
# Features:
#   
#   Energy
# 
# Entropy
# 
# Centroid frequency
# 
# Peak frequency
library(data.table)
library(caret)
library(smotefamily)
library(progress)
library(xgboost)

train = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
test = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")

subset_df <- train[, 6:ncol(train)]
smote1 <- data.frame(y = train$y, subset_df)

train_smote1 <- SMOTE(X = subset(smote1, y %in% c(1, 2)),
                      target = subset(smote1$y, smote1$y %in% c(1, 2)),
                      K = 38, dup_size = 0)

train_smote2 <- SMOTE(X = subset(smote1, y %in% c(1, 3)),
                      target = subset(smote1$y, smote1$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data)

data_plus <- train_smote_combined[!duplicated(train_smote_combined), ]
table(data_plus$y)

num_ts <- 24

new_data <- matrix(NA, nrow(data_plus), num_ts*4)

total_steps <- num_ts*nrow(data_plus)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data_plus)){
  for (j in 1:num_ts){
    ts <- as.numeric(data_plus[i, (1+(2048*(j-1))):(2049+(2048*(j-1)))])
    fourier_series <- fft(ts)
    new_data[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_data[i, (j+num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_data[i, (j+num_ts*2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_data[i, (j+num_ts*3)] <- which.max(abs(fourier_series)) # peak frequency
    
    pb$tick()
  }
}

new_data <- data.frame(new_data)
new_data$y <- data_plus$y

set.seed(123)

trainIndex <- createDataPartition(new_data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]

new_test <- matrix(NA, nrow(test), num_ts*4)

total_steps <- num_ts*nrow(new_test)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(test)){
  for (j in 1:num_ts){
    ts <- as.numeric(test[i, (5+(2048*(j-1))):(2052+(2048*(j-1)))])
    fourier_series <- fft(ts)
    new_test[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_test[i, (j+num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_test[i, (j+num_ts*2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_test[i, (j+num_ts*3)] <- which.max(abs(fourier_series)) # peak frequency
    pb$tick()
  }
}

train$y <- as.numeric(factor(train$y)) 

modello_xgboost <- xgboost(data = as.matrix(train[,-97]), 
                           label = train$y - 1, 
                           nrounds = 500,
                           objective = "multi:softprob", 
                           num_class = 3)

pred_probs <- predict(modello_xgboost, new_test, type = "class")
matrice_max <- matrix(nrow = length(pred_probs) %/% 3, ncol = 2)
for (i in 1:(length(pred_probs) %/% 3)) {
  gruppo <- pred_probs[(3 * (i - 1) + 1):(3 * i)]
  posizione_max <- which.max(gruppo)
  matrice_max[i, 1] <- max(gruppo)
  matrice_max[i, 2] <- posizione_max
}
risultato <- cbind(id = test$id, y = matrice_max[, 2])
table(risultato[,2])


# Different model better results, still not close to our goal...

# Attempt 3¶
# Model: glmnet
# 
# Oversampling: yes
# 
# Fourier: yes with fft
# 
# Features:
#   
#   Energy
# 
# Entropy
# 
# Centroid frequency
# 
# Peak frequency
# 
# THD (total harmonic distortion)
# 
# Crest factor
# 
# Spectral flatness
# 
# Spectral skewness
# 
# Spectral kurtosis

library(data.table)
library(glmnet)
library(smotefamily)
library(progress)
library(caret)

data = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
test = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")

subset_df <- data[, 6:ncol(data)]
smote1 <- data.frame(y = data$y, subset_df)

train_smote1 <- SMOTE(X = subset(smote1, y %in% c(1, 2)),
                      target = subset(smote1$y, smote1$y %in% c(1, 2)),
                      K = 3, dup_size = 0)

table(train_smote1$data$y)

train_smote2 <- SMOTE(X = subset(smote1, y %in% c(1, 3)),
                      target = subset(smote1$y, smote1$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

table(train_smote2$data$y)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data)subset_df <- data[, 6:ncol(data)]
smote1 <- data.frame(y = data$y, subset_df)

train_smote1 <- SMOTE(X = subset(smote1, y %in% c(1, 2)),
                      target = subset(smote1$y, smote1$y %in% c(1, 2)),
                      K = 3, dup_size = 0)

table(train_smote1$data$y)

train_smote2 <- SMOTE(X = subset(smote1, y %in% c(1, 3)),
                      target = subset(smote1$y, smote1$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

table(train_smote2$data$y)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data)

data_plus <- train_smote_combined[!duplicated(train_smote_combined), ]

num_ts <- 24

new_data <- matrix(NA, nrow(data_plus), num_ts*9)

total_steps <- num_ts*nrow(data_plus)
pb <- progress_bar$new(total = total_steps)

# Loop through each row of the data frame
for (i in 1:nrow(data_plus)) {
  # Loop through each time series
  for (j in 1:num_ts) {
    # Extract the time series data
    ts <- as.numeric(data_plus[i, (1 + (2048 * (j - 1))):(2049 + (2048 * (j - 1)))])
    
    sampling_rate <- 1
    n <- length(ts)
    frequency_resolution <- sampling_rate / n
    frequency_bins <- n/2 + 1
    
    # Perform FFT on the time series data
    fourier_series <- fft(ts)
    
    # Calculate the energy
    energy <- sum(abs(fourier_series)^2)
    
    # Calculate the entropy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    entropy <- -sum(normalized_series * log2(normalized_series))
    
    # Calculate the magnitude spectrum
    amplitude_spectrum <- Mod(fourier_series[1:frequency_bins])
    
    # Calculate the phase spectrum
    phase_spectrum <- Arg(fourier_series[1:frequency_bins])
    
    # Calculate the centroid frequency
    centroid_frequency <- sum(frequency_resolution * (1:frequency_bins) * amplitude_spectrum^2) / sum(amplitude_spectrum^2)
    
    # Calculate the peak frequency
    peak_frequency_index <- which.max(amplitude_spectrum)
    peak_frequency <- (peak_frequency_index - 1) * frequency_resolution
    
    # Calculate the total harmonic distortion (THD)
    fundamental_amplitude <- amplitude_spectrum[1]
    harmonics_amplitude <- amplitude_spectrum[-1]
    thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
    
    # Calculate the crest factor
    crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
    
    # Calculate the spectral flatness
    spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
    
    # Calculate the spectral skewness and kurtosis
    spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
    spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))^4) / (frequency_bins * sd(amplitude_spectrum)^4)
    
    # Store the extracted features in the new data frame
    new_data[i, j]              <- energy
    new_data[i, (j + num_ts)]   <- entropy
    new_data[i, (j + num_ts*2)] <- centroid_frequency
    new_data[i, (j + num_ts*3)] <- peak_frequency
    new_data[i, (j + num_ts*4)] <- thd
    new_data[i, (j + num_ts*5)] <- crest_factor
    new_data[i, (j + num_ts*6)] <- spectral_flatness
    new_data[i, (j + num_ts*7)] <- spectral_skewness
    new_data[i, (j + num_ts*8)] <- spectral_kurtosis
    
    pb$tick()
  }
}

new_data <- data.frame(new_data)
new_data$y <- data_plus$y
head(new_data)

set.seed(123)

trainIndex <- createDataPartition(new_data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]


model <- glmnet(as.matrix(train[, 1:(ncol(train)-1)]), train$y, lambda=0.1,
                maxit = 1e+09, alpha=0.5, family="multinomial")

predictions <- predict(model, as.matrix(validation[, 1:(ncol(validation)-1)]), type="class")

table(validation$y, predictions)

test <- data.frame(fread("test.csv"))
new_test <- matrix(NA, nrow(test), num_ts*9)

total_steps <- num_ts*nrow(new_test)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(test)){
  for (j in 1:num_ts){
    ts <- as.numeric(test[i, (5+(2048*(j-1))):(2052+(2048*(j-1)))])
    
    sampling_rate <- 1
    n <- length(ts)
    frequency_resolution <- sampling_rate / n
    frequency_bins <- n/2 + 1
    
    # Perform FFT on the time series data
    fourier_series <- fft(ts)
    
    # Calculate the energy
    energy <- sum(abs(fourier_series)^2)
    # Calculate the entropy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    entropy <- -sum(normalized_series * log2(normalized_series))
    
    # Calculate the magnitude spectrum
    amplitude_spectrum <- Mod(fourier_series[1:frequency_bins])
    
    # Calculate the phase spectrum
    phase_spectrum <- Arg(fourier_series[1:frequency_bins])
    
    # Calculate the centroid frequency
    centroid_frequency <- sum(frequency_resolution * (1:frequency_bins) * amplitude_spectrum^2) / sum(amplitude_spectrum^2)
    
    # Calculate the peak frequency
    peak_frequency_index <- which.max(amplitude_spectrum)
    peak_frequency <- (peak_frequency_index - 1) * frequency_resolution
    # Calculate the total harmonic distortion (THD)
    fundamental_amplitude <- amplitude_spectrum[1]
    harmonics_amplitude <- amplitude_spectrum[-1]
    thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
    
    # Calculate the crest factor
    crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
    
    # Calculate the spectral flatness
    spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
    
    # Calculate the spectral skewness and kurtosis
    spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
    spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))    # Calculate the total harmonic distortion (THD)
                             fundamental_amplitude <- amplitude_spectrum[1]
                             harmonics_amplitude <- amplitude_spectrum[-1]
                             thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
                             
                             # Calculate the crest factor
                             crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
                             
                             # Calculate the spectral flatness
                             spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
                             
                             # Calculate the spectral skewness and kurtosis
                             spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
                             spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))^4) / (frequency_bins * sd(amplitude_spectrum)^4)
                             
                                                      
                                                
                                                  

                             # Store the extracted features in the new data frame
                             new_test[i, j]              <- energy
                             new_test[i, (j + num_ts)]   <- entropy
                             new_test[i, (j + num_ts*2)] <- centroid_frequency
                             new_test[i, (j + num_ts*3)] <- peak_frequency
                             new_test[i, (j + num_ts*4)] <- thd
                             new_test[i, (j + num_ts*5)] <- crest_factor
                             new_test[i, (j + num_ts*6)] <- spectral_flatness
                             new_test[i, (j + num_ts*7)] <- spectral_skewness
                             
                             new_test[i, (j + num_ts*8)] <- spectral_kurtosis
                             
                             pb$tick()
  }
}
predictions <- predict(model, new_test, type="class")
table(predictions)
submission <- data.frame(id=test$id, y=as.numeric(predictions))
write.csv(submission, file = "sub6.csv", row.names = FALSE)


# Now we have more features, will it be enough? We can still add something...
# 
# Also in this case we tried different models and different parameters in order to find a good combination.
# 
# We also tried a features selection trough Random Forest but as you said this morning in the video call, it makes no sense to use features selected from one model and then use them in another (we realized that in the hard way).

    
#     
# Attempt 4
# Model: multinom (nnet)
# 
# Oversampling: no
# 
# Fourier: yes
# 
# Features:
#   
#   Energy
# 
# Entropy
# 
# Centroid frequency
# 
# Peak frequency
# 
# First element of Mod(fourier_series)
# 
# First element of Arg(fourier_series)




library(data.table)
library(progress)
library(nnet)

train = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
test = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")


num_ts <- 24
new_data <- matrix(NA, nrow(data), num_ts * 6)

total_steps <- num_ts * nrow(data)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data)) {
  for (j in 1:num_ts) {
    ts <- as.numeric(data[i, (5 + (2048 * (j - 1))):(2052 + (2048 * (j - 1)))])
    fourier_series <- fft(ts)
    new_data[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_data[i, (j + num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_data[i, (j + num_ts * 2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_data[i, (j + num_ts * 3)] <- which.max(abs(fourier_series)) # peak frequency
    
    new_data[i, (j + num_ts * 4)] <- Mod(fourier_series)[1] # using first element of Mod(fourier_series)
    new_data[i, (j + num_ts * 5)] <- Arg(fourier_series)[1] # using first element of Arg(fourier_series)
    pb$tick()
  }
}


new_data <- data.frame(new_data)
new_data$y <- data$y

set.seed(123)

trainIndex <- createDataPartition(new_data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]

test = test[, 5:49156]

num_ts <- 24
new_test <- matrix(NA, nrow(test), num_ts * 6)

total_steps <- num_ts * nrow(test)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(test)) {
  for (j in 1:num_ts) {
    ts_test <- as.numeric(test[i, ((2048 * (j - 1)) +1):((2048 * (j)))])
    fourier_series_test <- fft(ts_test)
    new_test[i, j] <- sum(abs(fourier_series_test)^2) # energy
    normalized_series_test <- abs(fourier_series_test) / sum(abs(fourier_series_test))
    new_test[i, (j + num_ts)] <- -sum(normalized_series_test * log2(normalized_series_test)) # entropy
    new_test[i, (j + num_ts * 2)] <- sum(normalized_series_test * seq(1, length(fourier_series_test))) # centroid frequency
    
    
    new_test[i, (j + num_ts * 3)] <- which.max(abs(fourier_series_test)) # peak frequency
    new_test[i, (j + num_ts * 4)] <- Mod(fourier_series_test)[1] # using first element of Mod(fourier_series)
    new_test[i, (j + num_ts * 5)] <- Arg(fourier_series_test)[1] # using first element of Arg(fourier_series)
    pb$tick()
  }
}

new_test <- data.frame(new_test)


new_data_subset = new_data[, -ncol(new_data)]
X <- as.matrix(new_data_subset)
y <- as.factor(new_data$y)

model <- multinom(y ~ ., data = data.frame(X),  maxit = 2750)
# Predictions
new_test <- as.matrix(new_test)
prediction <- predict(model, newdata = data.frame(new_test), type = "class")
table(prediction)


# 
# Here we are, the best model, at least on the public... uoOoOOoOu a logistic regression beats everyone (especially svm, we don't know why but we get only ones with svm no matter of the kernel).
# 
# In this attempt we have two more features:
# 
# First element of Mod(fourier_series)
# 
# First element of Arg(fourier_series)
# 
# Strange but somehow useful(?).
# 
# Variuos submissione are based on this model with some slightly changes (parameter, features, oversampling, ...).


# Attempt 5
# Model: NeuralNet (with Keras)
# 
# Oversampling: no
# 
# Fourier: yes
# 
# Features:
#   
#   Energy
# 
# Entropy
# 
# Centroid frequency
# 
# Peak frequency
# 
# THD (total harmonic distortion)
# 
# Crest factor
# 
# Spectral flatness
# 
# Spectral skewness
# 
# Spectral kurtosis

library(data.table)
library(progress)
library(keras)

data = data.frame(fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv"))
test = data.frame(fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv"))

num_ts <- 24

new_data <- matrix(NA, nrow(data), num_ts*9)

total_steps <- num_ts*nrow(data)
pb <- progress_bar$new(total = total_steps)

# Loop through each row of the data frame
for (i in 1:nrow(data)) {
  # Loop through each time series
  for (j in 1:num_ts) {
    # Extract the time series data
    ts <- as.numeric(data[i, (6 + (2048 * (j - 1))):(2053 + (2048 * (j - 1)))])
    
    sampling_rate <- 1
    n <- length(ts)
    frequency_resolution <- sampling_rate / n
    frequency_bins <- n/2 + 1
    
    # Perform FFT on the time series data
    fourier_series <- fft(ts)
    
    # Calculate the energy
    energy <- sum(abs(fourier_series)^2)
    
    # Calculate the entropy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    entropy <- -sum(normalized_series * log2(normalized_series))
    
    # Calculate the magnitude spectrum
    amplitude_spectrum <- Mod(fourier_series[1:frequency_bins])
    
    # Calculate the phase spectrum
    phase_spectrum <- Arg(fourier_series[1:frequency_bins])
    
    # Calculate the centroid frequency
    centroid_frequency <- sum(frequency_resolution * (1:frequency_bins) * amplitude_spectrum^2) / sum(amplitude_spectrum^2)
    
    # Calculate the peak frequency
    peak_frequency_index <- which.max(amplitude_spectrum)
    peak_frequency <- (peak_frequency_index - 1) * frequency_resolution
    
    # Calculate the total harmonic distortion (THD)
    fundamental_amplitude <- amplitude_spectrum[1]
    harmonics_amplitude <- amplitude_spectrum[-1]
    
    thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
    
    # Calculate the crest factor
    crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
    
    # Calculate the spectral flatness
    spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
    
    # Calculate the spectral skewness and kurtosis
    spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
    spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))^4) / (frequency_bins * sd(amplitude_spectrum)^4)
    
    # Store the extracted features in the new data frame
    new_data[i, j]              <- energy
    new_data[i, (j + num_ts)]   <- entropy
    new_data[i, (j + num_ts*2)] <- centroid_frequency
    new_data[i, (j + num_ts*3)] <- peak_frequency
    new_data[i, (j + num_ts*4)] <- thd
    new_data[i, (j + num_ts*5)] <- crest_factor
    new_data[i, (j + num_ts*6)] <- spectral_flatness
    new_data[i, (j + num_ts*7)] <- spectral_skewness
    new_data[i, (j + num_ts*8)] <- spectral_kurtosis
    
    pb$tick()
    
  }
}

new_data_ <- data.frame(new_data)
new_data_$y <- new_data_$y

est <- data.frame(fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv"))

num_ts <- 24

new_test <- matrix(NA, nrow(test), num_ts*9)

total_steps <- num_ts*nrow(test)
pb <- progress_bar$new(total = total_steps)

# Loop through each row of the data frame
for (i in 1:nrow(test)) {
  # Loop through each time series
  for (j in 1:num_ts) {
    # Extract the time series data
    ts <- as.numeric(test[i, (5 + (2048 * (j - 1))):(2052 + (2048 * (j - 1)))])
    
    sampling_rate <- 1
    n <- length(ts)
    frequency_resolution <- sampling_rate / n
    frequency_bins <- n/2 + 1
    
    # Perform FFT on the time series data
    fourier_series <- fft(ts)
    
    # Calculate the energy
    energy <- sum(abs(fourier_series)^2)
    
    # Calculate the entropy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    entropy <- -sum(normalized_series * log2(normalized_series))
    
    # Calculate the magnitude spectrum
    amplitude_spectrum <- Mod(fourier_series[1:frequency_bins])
    
    # Calculate the phase spectrum
    phase_spectrum <- Arg(fourier_series[1:frequency_bins])
    
    # Calculate the centroid frequency
    centroid_frequency <- sum(frequency_resolution * (1:frequency_bins) * amplitude_spectrum^2) / sum(amplitude_spectrum^2)
    
    # Calculate the peak frequency
    peak_frequency_index <- which.max(amplitude_spectrum)
    peak_frequency <- (peak_frequency_index - 1) * frequency_resolution
    
    # Calculate the total harmonic distortion (THD)
    
    fundamental_amplitude <- amplitude_spectrum[1]
    harmonics_amplitude <- amplitude_spectrum[-1]
    thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
    
    # Calculate the crest factor
    crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
    
    # Calculate the spectral flatness
    spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
    
    # Calculate the spectral skewness and kurtosis
    spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
    spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))^4) / (frequency_bins * sd(amplitude_spectrum)^4)
    
    
    # Store the extracted features in the new data frame
    new_test[i, j]              <- energy
    new_test[i, (j + num_ts)]   <- entropy
    new_test[i, (j + num_ts*2)] <- centroid_frequency
    new_test[i, (j + num_ts*3)] <- peak_frequency
    new_test[i, (j + num_ts*4)] <- thd
    new_test[i, (j + num_ts*5)] <- crest_factor
    new_test[i, (j + num_ts*6)] <- spectral_flatness
    new_test[i, (j + num_ts*7)] <- spectral_skewness
    new_test[i, (j + num_ts*8)] <- spectral_kurtosis
    
    pb$tick()
    
  }
}
new_test_ <- data.frame(new_test)

colnames(new_data_) <- c(paste0("X", 1:(ncol(new_data_)-1)), "y")
colnames(new_test_) <- paste0("X", 1:ncol(new_test_))



#Attempt 0.0
Model: glmnet

Oversampling: no

Fourier: no

Features: statistic of the boxtest applied to the autocorrelation series

Submitted: no

library(data.table)
library(glmnet)
library(progress)
library(caret)
Loading required package: Matrix

Loaded glmnet 4.1-7

Loading required package: ggplot2

Loading required package: lattice


Attaching package: ‘caret’


The following object is masked from ‘package:httr’:
  
  progress


data = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
num_ts <- 24

new_data <- matrix(NA, nrow(data), 24)

total_steps <- 24*nrow(data)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data)){
  for (j in 1:24){
    ts <- as.numeric(data[i, (5+(2048*(j-1))):(2053+(2048*(j-1)))])
    ljung_box_test <- Box.test(ts, lag = 10, type = "Ljung-Box")$statistic
    new_data[i, j] <- as.numeric(ljung_box_test)
    pb$tick()
  }
}
new_data <- data.frame(new_data)
new_data$y <- data$y
head(new_data)
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]

model_1 <- glmnet(as.matrix(train[, 1:(ncol(train)-1)]), train$y, lambda=0.0002, alpha=0, family="multinomial")
predictions_1 <- predict(model_1, as.matrix(validation[, 1:(ncol(validation)-1)]), type="class")

table(validation$y, predictions_1)
Comment
We tried different combination of parameters for glmnet and also different models but non of them were convincing.

Attempt 0.1
Model: glmnet

Oversampling: no

Fourier: no

Features:
  
  sd

median

iqr

diff(range))
    
    Submitted: no
    
    library(data.table)
    library(glmnet)
    library(progress)
    library(caret)
    data = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
    num_ts <- 24 # number of time series
    
    new_data <- matrix(NA, nrow(data), num_ts*4) # new dataset based on the features
    
    total_steps <- num_ts*nrow(data) 
    pb <- progress_bar$new(total = total_steps)
    
    for (i in 1:nrow(data)){
      for (j in 1:num_ts){
        ts <- as.numeric(data[i, (5+(2048*(j-1))):(2053+(2048*(j-1)))])
        new_data[i, j] <- median(ts)
        new_data[i, (j+num_ts)] <- sd(ts)
        new_data[i, (j+num_ts*2)] <- IQR(ts)
        new_data[i, (j+num_ts*3)] <- diff(range(ts))
        pb$tick()
      }
    }
    new_data <- data.frame(new_data)
    new_data$y <- data$y
    head(new_data)
    Create train and validation set and see the result on the validation
    
    trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
    
    train <- new_data[trainIndex, ]
    validation <- new_data[-trainIndex, ]
    
    model <- glmnet(as.matrix(train[, 1:(ncol(train)-1)]), train$y, lambda=0.003, alpha=0.5, family="multinomial")
    predictions <- predict(model, as.matrix(validation[, 1:(ncol(validation)-1)]), type="class")
    
    table(validation$y, predictions)
    Comment
    We tried different combination of parameters for glmnet and also different models but non of them were convincing.
    
    Attempt 1
    Model: glmnet
    
    Oversampling: yes
    
    Fourier: until 50 and then removed elements with abs(x) < 0.01
    
    Features: Fourier series coefficients
    
    Submitted: yes
    
    "sub1.csv"
    library(data.table)
    library(glmnet)
    library(smotefamily)
    gino = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
    mat = as.matrix(gino[,6:49157])
    mat_fou <- matrix(NA,663,(101)*24)
    
    N_data <- length(mat[1,1:2048])
    N <- 50
    
    for(i in 1:663){
      
      for(j in 1:24){
        data <- mat[i,((j-1)*2048+1):(j*2048)]
        
        a0 <- sum(data)/N_data
        
        an <- numeric(N)
        bn <- numeric(N)
        
        for (k in 1:N) {
          an[k] <- 2 * sum(data * cos(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
          bn[k] <- 2 * sum(data * sin(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
        }
        
        mat_fou[i,((j-1)*101+1):(j*101)] <- c(a0,an,bn)
      }
    }
    dat_fou <- data.frame(mat_fou)
    dat_fou$y <- gino$y
    
    train_smote1 <- SMOTE(X = subset(dat_fou, y %in% c(1, 2)),
                          target = subset(dat_fou$y, dat_fou$y %in% c(1, 2)),
                          K = 3, dup_size = 0)
    
    table(train_smote1$data$y)
    train_smote2 <- SMOTE(X = subset(dat_fou, y %in% c(1, 3)),
                          target = subset(dat_fou$y, dat_fou$y %in% c(1, 3)),
                          K = 3, dup_size = 0)
    
    table(train_smote2$data$y)
    train_smote_combined <- rbind(train_smote1$data, train_smote2$data[train_smote2$data$y==3,])[,-2426]
    table(train_smote_combined$y)
    sam <- sample(dim(train_smote_combined)[1],1400, replace=F)
    
    pino <- cv.glmnet(as.matrix(train_smote_combined[sam,-2425]), train_smote_combined[sam,2425], family="multinomial")
    pino <- glmnet(train_smote_combined[,-2425], train_smote_combined[,2425],
                   family="multinomial", alpha=1, lambda=0.003705)
    test <- fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")
    mat_test = as.matrix(test[,5:49156])
    mat_fou_test <- matrix(NA,355,(101)*24)
    
    N_data <- length(mat[1,1:2048])
    N <- 50
    
    for(i in 1:355){
      print(i)
      for(j in 1:24){
        data <- mat_test[i,((j-1)*2048+1):(j*2048)]
        
        a0 <- sum(data)/N_data
        
        an <- numeric(N)
        bn <- numeric(N)
        
        for (k in 1:N) {
          an[k] <- 2 * sum(data * cos(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
          bn[k] <- 2 * sum(data * sin(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
        }
        
        mat_fou_test[i,((j-1)*101+1):(j*101)] <- c(a0,an,bn)
      }
    }
    predictions <- predict(pino, mat_fou_test, type="class")
    submission <- data.frame(id=test$id, y=as.numeric(predictions))
    write.csv(submission, file = "sub1.csv", row.names = FALSE)
    Comment
    First simple attempt with Fourier... it could have been better.
    
    We tried different model and parameters but we didn't find a good combination so we started focusing on features engineering.

Attempt 2
Model: glmnet

Oversampling: yes

Fourier: yes with fft

Features:

Energy

Entropy

Centroid frequency

Peak frequency

Submitted: yes

"sub3.csv" (...probably...)
library(data.table)
library(glmnet)
library(smotefamily)
library(progress)
library(caret)
Oversampling with SMOTE
subset_df <- data[, 6:ncol(data)]
smote1 <- data.frame(y = data$y, subset_df)

train_smote1 <- SMOTE(X = subset(smote1, y %in% c(1, 2)),
                      target = subset(smote1$y, smote1$y %in% c(1, 2)),
                      K = 3, dup_size = 0)

table(train_smote1$data$y)

train_smote2 <- SMOTE(X = subset(smote1, y %in% c(1, 3)),
                      target = subset(smote1$y, smote1$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

table(train_smote2$data$y)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data)

data_plus <- train_smote_combined[!duplicated(train_smote_combined), ]
Fourier & Features extraction for train set
num_ts <- 24

new_data <- matrix(NA, nrow(data_plus), num_ts*4)

total_steps <- num_ts*nrow(data_plus)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data_plus)){
  for (j in 1:num_ts){
    ts <- as.numeric(data_plus[i, (1+(2048*(j-1))):(2049+(2048*(j-1)))])
    fourier_series <- fft(ts)
    new_data[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_data[i, (j+num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_data[i, (j+num_ts*2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_data[i, (j+num_ts*3)] <- which.max(abs(fourier_series)) # peak frequency
    pb$tick()
  }
}

new_data <- data.frame(new_data)
new_data$y <- data_plus$y
head(new_data)
Partitioning the dataset
set.seed(123)

trainIndex <- createDataPartition(new_data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]
Model
model <- glmnet(as.matrix(train[, 1:(ncol(train)-1)]), train$y, lambda=0.001,
                maxit = 1e+09, alpha=1, family="multinomial")

predictions <- predict(model, as.matrix(validation[, 1:(ncol(validation)-1)]), type="class")

table(validation$y, predictions)
Fourier & Features extraction for test set
test <- fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")
new_test <- matrix(NA, nrow(test), num_ts*4)

total_steps <- num_ts*nrow(new_test)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(test)){
  for (j in 1:num_ts){
    ts <- as.numeric(test[i, (5+(2048*(j-1))):(2052+(2048*(j-1)))])
    fourier_series <- fft(ts)
    new_test[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_test[i, (j+num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_test[i, (j+num_ts*2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_test[i, (j+num_ts*3)] <- which.max(abs(fourier_series)) # peak frequency
    pb$tick()
  }
}
Predictions on test set
predictions <- predict(model, new_test, type="class")
table(predictions)
submission <- data.frame(id=test$id, y=as.numeric(predictions))
write.csv(submission, file = "sub3.csv", row.names = FALSE)
Comment
Now we have features, will it be enough? Not yet.

Also in this case we tried different models and different parameters, one surprised us (in the following subs...)

Attempt 2.1
Model: Gradient Boosting

Oversampling: yes

Fourier: yes

Features:

Energy

Entropy

Centroid frequency

Peak frequency

Submitted: yes

"risultato.csv" (...should be...)
library(data.table)
library(caret)
library(smotefamily)
library(progress)
library(xgboost)

train = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
test = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")
Oversampling with SMOTE
subset_df <- train[, 6:ncol(train)]
smote1 <- data.frame(y = train$y, subset_df)

train_smote1 <- SMOTE(X = subset(smote1, y %in% c(1, 2)),
                      target = subset(smote1$y, smote1$y %in% c(1, 2)),
                      K = 38, dup_size = 0)

train_smote2 <- SMOTE(X = subset(smote1, y %in% c(1, 3)),
                      target = subset(smote1$y, smote1$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data)

data_plus <- train_smote_combined[!duplicated(train_smote_combined), ]
table(data_plus$y)
Fourier & Features extraction for train set
num_ts <- 24

new_data <- matrix(NA, nrow(data_plus), num_ts*4)

total_steps <- num_ts*nrow(data_plus)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data_plus)){
  for (j in 1:num_ts){
    ts <- as.numeric(data_plus[i, (1+(2048*(j-1))):(2049+(2048*(j-1)))])
    fourier_series <- fft(ts)
    new_data[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_data[i, (j+num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_data[i, (j+num_ts*2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_data[i, (j+num_ts*3)] <- which.max(abs(fourier_series)) # peak frequency
    pb$tick()
  }
}

new_data <- data.frame(new_data)
new_data$y <- data_plus$y
Partitioning
set.seed(123)

trainIndex <- createDataPartition(new_data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]
Fourier & Features extraction for test set
new_test <- matrix(NA, nrow(test), num_ts*4)

total_steps <- num_ts*nrow(new_test)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(test)){
  for (j in 1:num_ts){
    ts <- as.numeric(test[i, (5+(2048*(j-1))):(2052+(2048*(j-1)))])
    fourier_series <- fft(ts)
    new_test[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_test[i, (j+num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_test[i, (j+num_ts*2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_test[i, (j+num_ts*3)] <- which.max(abs(fourier_series)) # peak frequency
    pb$tick()
  }
}
Model
train$y <- as.numeric(factor(train$y)) 

modello_xgboost <- xgboost(data = as.matrix(train[,-97]), 
                           label = train$y - 1, 
                           nrounds = 500,
                           objective = "multi:softprob", 
                           num_class = 3)
Predictions
pred_probs <- predict(modello_xgboost, new_test, type = "class")
matrice_max <- matrix(nrow = length(pred_probs) %/% 3, ncol = 2)
for (i in 1:(length(pred_probs) %/% 3)) {
  gruppo <- pred_probs[(3 * (i - 1) + 1):(3 * i)]
  posizione_max <- which.max(gruppo)
  matrice_max[i, 1] <- max(gruppo)
  matrice_max[i, 2] <- posizione_max
}
risultato <- cbind(id = test$id, y = matrice_max[, 2])
table(risultato[,2])
write.csv(risultato, file = "risultato.csv", row.names = FALSE)
Comment
Different model better results, still not close to our goal...

Attempt 3
Model: glmnet

Oversampling: yes

Fourier: yes with fft

Features:

Energy

Entropy

Centroid frequency

Peak frequency

THD (total harmonic distortion)

Crest factor

Spectral flatness

Spectral skewness

Spectral kurtosis

Submitted: yes

"sub6.csv" (...probably...)
library(data.table)
library(glmnet)
library(smotefamily)
library(progress)
library(caret)

data = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
test = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")
Oversampling with SMOTE
subset_df <- data[, 6:ncol(data)]
smote1 <- data.frame(y = data$y, subset_df)

train_smote1 <- SMOTE(X = subset(smote1, y %in% c(1, 2)),
                      target = subset(smote1$y, smote1$y %in% c(1, 2)),
                      K = 3, dup_size = 0)

table(train_smote1$data$y)

train_smote2 <- SMOTE(X = subset(smote1, y %in% c(1, 3)),
                      target = subset(smote1$y, smote1$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

table(train_smote2$data$y)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data)

data_plus <- train_smote_combined[!duplicated(train_smote_combined), ]
Fourier & Features extraction for train set
num_ts <- 24

new_data <- matrix(NA, nrow(data_plus), num_ts*9)

total_steps <- num_ts*nrow(data_plus)
pb <- progress_bar$new(total = total_steps)

# Loop through each row of the data frame
for (i in 1:nrow(data_plus)) {
  # Loop through each time series
  for (j in 1:num_ts) {
    # Extract the time series data
    ts <- as.numeric(data_plus[i, (1 + (2048 * (j - 1))):(2049 + (2048 * (j - 1)))])
    
    sampling_rate <- 1
    n <- length(ts)
    frequency_resolution <- sampling_rate / n
    frequency_bins <- n/2 + 1
    
    # Perform FFT on the time series data
    fourier_series <- fft(ts)

    # Calculate the energy
    energy <- sum(abs(fourier_series)^2)
    
    # Calculate the entropy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    entropy <- -sum(normalized_series * log2(normalized_series))
        
    # Calculate the magnitude spectrum
    amplitude_spectrum <- Mod(fourier_series[1:frequency_bins])
    
    # Calculate the phase spectrum
    phase_spectrum <- Arg(fourier_series[1:frequency_bins])
    
    # Calculate the centroid frequency
    centroid_frequency <- sum(frequency_resolution * (1:frequency_bins) * amplitude_spectrum^2) / sum(amplitude_spectrum^2)
    
    # Calculate the peak frequency
    peak_frequency_index <- which.max(amplitude_spectrum)
    peak_frequency <- (peak_frequency_index - 1) * frequency_resolution
    
    # Calculate the total harmonic distortion (THD)
    fundamental_amplitude <- amplitude_spectrum[1]
    harmonics_amplitude <- amplitude_spectrum[-1]
    thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
    
    # Calculate the crest factor
    crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
    
    # Calculate the spectral flatness
    spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
    
    # Calculate the spectral skewness and kurtosis
    spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
    spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))^4) / (frequency_bins * sd(amplitude_spectrum)^4)
    
    # Store the extracted features in the new data frame
    new_data[i, j]              <- energy
    new_data[i, (j + num_ts)]   <- entropy
    new_data[i, (j + num_ts*2)] <- centroid_frequency
    new_data[i, (j + num_ts*3)] <- peak_frequency
    new_data[i, (j + num_ts*4)] <- thd
    new_data[i, (j + num_ts*5)] <- crest_factor
    new_data[i, (j + num_ts*6)] <- spectral_flatness
    new_data[i, (j + num_ts*7)] <- spectral_skewness
    new_data[i, (j + num_ts*8)] <- spectral_kurtosis

    pb$tick()
  }
}
new_data <- data.frame(new_data)
new_data$y <- data_plus$y
head(new_data)
Create train & validation sets
set.seed(123)

trainIndex <- createDataPartition(new_data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]
Model
model <- glmnet(as.matrix(train[, 1:(ncol(train)-1)]), train$y, lambda=0.1,
                maxit = 1e+09, alpha=0.5, family="multinomial")

predictions <- predict(model, as.matrix(validation[, 1:(ncol(validation)-1)]), type="class")

table(validation$y, predictions)
Fourier & Features extraction for test set
test <- data.frame(fread("test.csv"))
new_test <- matrix(NA, nrow(test), num_ts*9)

total_steps <- num_ts*nrow(new_test)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(test)){
  for (j in 1:num_ts){
    ts <- as.numeric(test[i, (5+(2048*(j-1))):(2052+(2048*(j-1)))])
    
    sampling_rate <- 1
    n <- length(ts)
    frequency_resolution <- sampling_rate / n
    frequency_bins <- n/2 + 1
    
    # Perform FFT on the time series data
    fourier_series <- fft(ts)
    
    # Calculate the energy
    energy <- sum(abs(fourier_series)^2)
    
    # Calculate the entropy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    entropy <- -sum(normalized_series * log2(normalized_series))
    
    # Calculate the magnitude spectrum
    amplitude_spectrum <- Mod(fourier_series[1:frequency_bins])
    
    # Calculate the phase spectrum
    phase_spectrum <- Arg(fourier_series[1:frequency_bins])
    
    # Calculate the centroid frequency
    centroid_frequency <- sum(frequency_resolution * (1:frequency_bins) * amplitude_spectrum^2) / sum(amplitude_spectrum^2)
    
    # Calculate the peak frequency
    peak_frequency_index <- which.max(amplitude_spectrum)
    peak_frequency <- (peak_frequency_index - 1) * frequency_resolution
    
    # Calculate the total harmonic distortion (THD)
    fundamental_amplitude <- amplitude_spectrum[1]
    harmonics_amplitude <- amplitude_spectrum[-1]
    thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
    
    # Calculate the crest factor
    crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
    
    # Calculate the spectral flatness
    spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
    
    # Calculate the spectral skewness and kurtosis
    spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
    spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))^4) / (frequency_bins * sd(amplitude_spectrum)^4)
    
    # Store the extracted features in the new data frame
    new_test[i, j]              <- energy
    new_test[i, (j + num_ts)]   <- entropy
    new_test[i, (j + num_ts*2)] <- centroid_frequency
    new_test[i, (j + num_ts*3)] <- peak_frequency
    new_test[i, (j + num_ts*4)] <- thd
    new_test[i, (j + num_ts*5)] <- crest_factor
    new_test[i, (j + num_ts*6)] <- spectral_flatness
    new_test[i, (j + num_ts*7)] <- spectral_skewness
    new_test[i, (j + num_ts*8)] <- spectral_kurtosis
    
    pb$tick()
  }
}
Predictions
predictions <- predict(model, new_test, type="class")
table(predictions)
submission <- data.frame(id=test$id, y=as.numeric(predictions))
write.csv(submission, file = "sub6.csv", row.names = FALSE)
Comment
Now we have more features, will it be enough? We can still add something...

Also in this case we tried different models and different parameters in order to find a good combination.

We also tried a features selection trough Random Forest but as you said this morning in the video call, it makes no sense to use features selected from one model and then use them in another (we realized that in the hard way).

Attempt 4
Model: multinom (nnet)

Oversampling: no

Fourier: yes

Features:

Energy

Entropy

Centroid frequency

Peak frequency

First element of Mod(fourier_series)

First element of Arg(fourier_series)

Submitted: yes

"sub16.csv"

"sub15.csv" (just using 2500 iterations instead of the 2750 reported later)

"sub29.csv" (just using 3000 iterations and all the 9 features instead of 4)

library(data.table)
library(progress)
library(nnet)

train = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
test = fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")
Fourier & Features extraction for train set
num_ts <- 24
new_data <- matrix(NA, nrow(data), num_ts * 6)

total_steps <- num_ts * nrow(data)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(data)) {
  for (j in 1:num_ts) {
    ts <- as.numeric(data[i, (5 + (2048 * (j - 1))):(2052 + (2048 * (j - 1)))])
    fourier_series <- fft(ts)
    new_data[i, j] <- sum(abs(fourier_series)^2) # energy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    new_data[i, (j + num_ts)] <- -sum(normalized_series * log2(normalized_series)) # entropy
    new_data[i, (j + num_ts * 2)] <- sum(normalized_series * seq(1, length(fourier_series))) # centroid frequency
    new_data[i, (j + num_ts * 3)] <- which.max(abs(fourier_series)) # peak frequency
    new_data[i, (j + num_ts * 4)] <- Mod(fourier_series)[1] # using first element of Mod(fourier_series)
    new_data[i, (j + num_ts * 5)] <- Arg(fourier_series)[1] # using first element of Arg(fourier_series)
    pb$tick()
  }
}


new_data <- data.frame(new_data)
new_data$y <- data$y
Partitioning
set.seed(123)

trainIndex <- createDataPartition(new_data$y, p = 0.8, list = FALSE)

train <- new_data[trainIndex, ]
validation <- new_data[-trainIndex, ]
Fourier & Features extraction for test set
test = test[, 5:49156]

num_ts <- 24
new_test <- matrix(NA, nrow(test), num_ts * 6)

total_steps <- num_ts * nrow(test)
pb <- progress_bar$new(total = total_steps)

for (i in 1:nrow(test)) {
  for (j in 1:num_ts) {
    ts_test <- as.numeric(test[i, ((2048 * (j - 1)) +1):((2048 * (j)))])
    fourier_series_test <- fft(ts_test)
    new_test[i, j] <- sum(abs(fourier_series_test)^2) # energy
    normalized_series_test <- abs(fourier_series_test) / sum(abs(fourier_series_test))
    new_test[i, (j + num_ts)] <- -sum(normalized_series_test * log2(normalized_series_test)) # entropy
    new_test[i, (j + num_ts * 2)] <- sum(normalized_series_test * seq(1, length(fourier_series_test))) # centroid frequency
    new_test[i, (j + num_ts * 3)] <- which.max(abs(fourier_series_test)) # peak frequency
    new_test[i, (j + num_ts * 4)] <- Mod(fourier_series_test)[1] # using first element of Mod(fourier_series)
    new_test[i, (j + num_ts * 5)] <- Arg(fourier_series_test)[1] # using first element of Arg(fourier_series)
    pb$tick()
  }
}

new_test <- data.frame(new_test)
Model
new_data_subset = new_data[, -ncol(new_data)]
X <- as.matrix(new_data_subset)
y <- as.factor(new_data$y)

model <- multinom(y ~ ., data = data.frame(X),  maxit = 2750)
Predictions
new_test <- as.matrix(new_test)
prediction <- predict(model, newdata = data.frame(new_test), type = "class")
table(prediction)
submission <- data.frame(id=test_completo$id, y=as.numeric(prediction))
write.csv(submission, file = "sub16.csv", row.names = FALSE)
Comment
Here we are, the best model, at least on the public... uoOoOOoOu a logistic regression beats everyone (especially svm, we don't know why but we get only ones with svm no matter of the kernel).

In this attempt we have two more features:
  
  First element of Mod(fourier_series)

First element of Arg(fourier_series)

Strange but somehow useful(?).

Variuos submissione are based on this model with some slightly changes (parameter, features, oversampling, ...).

Attempt 5
Model: NeuralNet (with Keras)

Oversampling: no

Fourier: yes

Features:
  
  Energy

Entropy

Centroid frequency

Peak frequency

THD (total harmonic distortion)

Crest factor

Spectral flatness

Spectral skewness

Spectral kurtosis

Submitted yes

"NNpower.csv"
library(data.table)
library(progress)
library(keras)

data = data.frame(fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv"))
test = data.frame(fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv"))
Fourier & Features extraction for train set
num_ts <- 24

new_data <- matrix(NA, nrow(data), num_ts*9)

total_steps <- num_ts*nrow(data)
pb <- progress_bar$new(total = total_steps)

# Loop through each row of the data frame
for (i in 1:nrow(data)) {
  # Loop through each time series
  for (j in 1:num_ts) {
    # Extract the time series data
    ts <- as.numeric(data[i, (6 + (2048 * (j - 1))):(2053 + (2048 * (j - 1)))])
    
    sampling_rate <- 1
    n <- length(ts)
    frequency_resolution <- sampling_rate / n
    frequency_bins <- n/2 + 1
    
    # Perform FFT on the time series data
    fourier_series <- fft(ts)
    
    # Calculate the energy
    energy <- sum(abs(fourier_series)^2)
    
    # Calculate the entropy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    entropy <- -sum(normalized_series * log2(normalized_series))
    
    # Calculate the magnitude spectrum
    amplitude_spectrum <- Mod(fourier_series[1:frequency_bins])
    
    # Calculate the phase spectrum
    phase_spectrum <- Arg(fourier_series[1:frequency_bins])
    
    # Calculate the centroid frequency
    centroid_frequency <- sum(frequency_resolution * (1:frequency_bins) * amplitude_spectrum^2) / sum(amplitude_spectrum^2)
    
    # Calculate the peak frequency
    peak_frequency_index <- which.max(amplitude_spectrum)
    peak_frequency <- (peak_frequency_index - 1) * frequency_resolution
    
    # Calculate the total harmonic distortion (THD)
    fundamental_amplitude <- amplitude_spectrum[1]
    harmonics_amplitude <- amplitude_spectrum[-1]
    thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
    
    # Calculate the crest factor
    crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
    
    # Calculate the spectral flatness
    spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
    
    # Calculate the spectral skewness and kurtosis
    spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
    spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))^4) / (frequency_bins * sd(amplitude_spectrum)^4)
    
    # Store the extracted features in the new data frame
    new_data[i, j]              <- energy
    new_data[i, (j + num_ts)]   <- entropy
    new_data[i, (j + num_ts*2)] <- centroid_frequency
    new_data[i, (j + num_ts*3)] <- peak_frequency
    new_data[i, (j + num_ts*4)] <- thd
    new_data[i, (j + num_ts*5)] <- crest_factor
    new_data[i, (j + num_ts*6)] <- spectral_flatness
    new_data[i, (j + num_ts*7)] <- spectral_skewness
    new_data[i, (j + num_ts*8)] <- spectral_kurtosis
    
    pb$tick()
    
  }
}


new_data_ <- data.frame(new_data)
new_data_$y <- new_data_$y
Fourier & Features extraction for test set
test <- data.frame(fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv"))

num_ts <- 24

new_test <- matrix(NA, nrow(test), num_ts*9)

total_steps <- num_ts*nrow(test)
pb <- progress_bar$new(total = total_steps)

# Loop through each row of the data frame
for (i in 1:nrow(test)) {
  # Loop through each time series
  for (j in 1:num_ts) {
    # Extract the time series data
    ts <- as.numeric(test[i, (5 + (2048 * (j - 1))):(2052 + (2048 * (j - 1)))])
    
    sampling_rate <- 1
    n <- length(ts)
    frequency_resolution <- sampling_rate / n
    frequency_bins <- n/2 + 1
    
    # Perform FFT on the time series data
    fourier_series <- fft(ts)
    
    # Calculate the energy
    energy <- sum(abs(fourier_series)^2)
    
    # Calculate the entropy
    normalized_series <- abs(fourier_series) / sum(abs(fourier_series))
    entropy <- -sum(normalized_series * log2(normalized_series))
    
    # Calculate the magnitude spectrum
    amplitude_spectrum <- Mod(fourier_series[1:frequency_bins])
    
    # Calculate the phase spectrum
    phase_spectrum <- Arg(fourier_series[1:frequency_bins])
    
    # Calculate the centroid frequency
    centroid_frequency <- sum(frequency_resolution * (1:frequency_bins) * amplitude_spectrum^2) / sum(amplitude_spectrum^2)
    
    # Calculate the peak frequency
    peak_frequency_index <- which.max(amplitude_spectrum)
    peak_frequency <- (peak_frequency_index - 1) * frequency_resolution
    
    # Calculate the total harmonic distortion (THD)
    fundamental_amplitude <- amplitude_spectrum[1]
    harmonics_amplitude <- amplitude_spectrum[-1]
    thd <- sqrt(sum(harmonics_amplitude^2)) / fundamental_amplitude
    
    # Calculate the crest factor
    crest_factor <- max(amplitude_spectrum) / sqrt(mean(ts^2))
    
    # Calculate the spectral flatness
    spectral_flatness <- exp(mean(log(amplitude_spectrum))) / mean(amplitude_spectrum)
    
    # Calculate the spectral skewness and kurtosis
    spectral_skewness <- sum((amplitude_spectrum - mean(amplitude_spectrum))^3) / (frequency_bins * sd(amplitude_spectrum)^3)
    spectral_kurtosis <- sum((amplitude_spectrum - mean(amplitude_spectrum))^4) / (frequency_bins * sd(amplitude_spectrum)^4)
    
    # Store the extracted features in the new data frame
    new_test[i, j]              <- energy
    new_test[i, (j + num_ts)]   <- entropy
    new_test[i, (j + num_ts*2)] <- centroid_frequency
    new_test[i, (j + num_ts*3)] <- peak_frequency
    new_test[i, (j + num_ts*4)] <- thd
    new_test[i, (j + num_ts*5)] <- crest_factor
    new_test[i, (j + num_ts*6)] <- spectral_flatness
    new_test[i, (j + num_ts*7)] <- spectral_skewness
    new_test[i, (j + num_ts*8)] <- spectral_kurtosis
    
    pb$tick()
    
  }
}

new_test_ <- data.frame(new_test)

colnames(new_data_) <- c(paste0("X", 1:(ncol(new_data_)-1)), "y")
colnames(new_test_) <- paste0("X", 1:ncol(new_test_))

#Scaling everything
col_to_remove <- c("X75", "X79", "X83", "X87", "X91", "X95") # contains same values in all the rows

new_data_ <- new_data_[, -which(names(new_data_) %in% col_to_remove)]
new_test_ <- new_test_[, -which(names(new_test_) %in% col_to_remove)]

train_scaled  <- data.frame(cbind(scale(new_data_[, 1:(ncol(new_data_)-1)]), y=new_data_$y))
test_scaled <- data.frame(scale(new_test_))
model <- keras_model_sequential()

model %>%
  layer_dense(units=200, input_shape=210) %>%
  layer_activation_softmax() %>%
  layer_dense(units=100) %>%
  layer_activation_softmax() %>%
  layer_dense(units=3) %>%
  layer_activation_softmax()

compile(model, optimizer="adam", loss="categorical_crossentropy")

fit(model, as.matrix(new_data_[,-211]), to_categorical(new_data_[,211]-1), epochs = 500)
predictions <- max.col(predict(model, as.matrix(new_test_)))
table(predictions)

# Attempt 6¶
# Model: CNN (with Keras)
# 
# Oversampling: yes
# 
# Fourier: yes
# 
# Features:
#   
#   Coefficients till k=100
# 
# Removed if lower than 0.01
# 
# Submitted no
# 
# In this attempt we tried rebuild the data structure through the Fourier series, keeping the values until K = 100 and throw away all the items with a coefficient to small considering them as noise (small means coefficients whit an absolute value less then 0.01). We've done the Fourier series from scratch: why? 

library(data.table)
library(glmnet)
library(smotefamily)
library(nnet)
library(keras)
library(reticulate)
library(boot)
library(caret)

train <- fread("/kaggle/input/statistical-learning-sapienza-spring-2023/train.csv")
test <- as.matrix(fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")[,5:49156])
t <- fread("/kaggle/input/statistical-learning-sapienza-spring-2023/test.csv")
mat = as.matrix(train[,6:49157])

# Number of elements in the data
N_data <- length(mat[1,1:2048])
# Number of elements in the series
N <- 100

mat_fou_coe <- matrix(NA,663,201*24)
mat_fou_val <- matrix(NA,663,2048*24)

### Build train series
for(i in 1:663){
  print(i)
  for(j in 1:24){
    data <- mat[i,((j-1)*2048+1):(j*2048)]
    
    a0 <- sum(data)/N_data
    
    an <- numeric(N)
    bn <- numeric(N)
    
    for (k in 1:N) {
      a <- 2 * sum(data * cos(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
      an[k] <- ifelse(abs(a)>0.01,a,0)
      b <- 2 * sum(data * sin(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
      bn[k] <- ifelse(abs(b)>0.01,b,0)
    }
    mat_fou_val[i,((j-1)*2048+1):(j*2048)] <- rep(a0,2048)+
      colSums(t(sapply(1:N, function(k) an[k]*cos(2 * pi * (k-1) * (0:(N_data-1)) / N_data) + bn[k]*sin(2 * pi * (k-1) * (0:(N_data-1)) / N_data))))
    mat_fou_coe[i,((j-1)*201+1):(j*201)] <- c(a0,an,bn)
  }
}

#########################################
mat_fou_test_coe <- matrix(NA,355,(201)*24)
mat_fou_test_val <- matrix(NA,355,(2048)*24)

for(i in 1:355){
  print(i)
  for(j in 1:24){
    data <- as.numeric(t[i,((j-1)*2048+5):((j*2048)+4)])
    
    a0 <- sum(data)/N_data
    
    an <- numeric(N)
    bn <- numeric(N)
    
    for (k in 1:N) {
      
      a <- 2 * sum(data * cos(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
      an[k] <- ifelse(abs(a)>0.01,a,0)
      b <- 2 * sum(data * sin(2 * pi * (k-1) * (0:(N_data-1)) / N_data)) / N_data
      bn[k] <- ifelse(abs(b)>0.01,b,0)
    }
    
    mat_fou_test_val[i,((j-1)*2048+1):(j*2048)] <- rep(a0,2048) + colSums(t(sapply(1:N, function(k) an[k]*cos(2 * pi * (k-1) * (0:(N_data-1)) / N_data) + bn[k]*sin(2 * pi * (k-1) * (0:(N_data-1)) / N_data))))
    
    mat_fou_test_coe[i,((j-1)*201+1):(j*201)] <- c(a0,an,bn)
  }
}



# OVERSAMPLING
dat_fou <- data.frame(mat_fou_val)
dat_fou$y <- train$y

train_smote1 <- SMOTE(X = subset(dat_fou, y %in% c(1, 2)),
                      target = subset(dat_fou$y, dat_fou$y %in% c(1, 2)),
                      K = 3, dup_size = 0)

table(train_smote1$data$y)

train_smote2 <- SMOTE(X = subset(dat_fou, y %in% c(1, 3)),
                      target = subset(dat_fou$y, dat_fou$y %in% c(1, 3)),
                      K = 3, dup_size = 0)

table(train_smote2$data$y)

train_smote_combined <- rbind(train_smote1$data, train_smote2$data[train_smote2$data$y==3,])

class <- train_smote_combined$y

aaa_res <- array_reshape(as.matrix(train_smote_combined[,-c(49153,49154)]), c(nrow(train_smote_combined[,-c(49153,49154)]), 24, 2048, 1))
aaa_test <- array_reshape(as.matrix(mat_fou_test_val), c(nrow(mat_fou_test_val), 24, 2048, 1))
# 
# This part it's a bit difficult to explain: someone could say "if in a compressor occurs a problem probably it may be some picks or something strange in a precise moment respect to nearest moments", so we use filters which has 24 rows (all the features) and  
# 4,5, 10 columns, obtaing each time a vector. With this technique we would measure in some sense a "relationship" between the "nearest" moments. It seems very dumb, indeed it is.

cnn_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 16, kernel_size = c(24,4), activation = 'relu', input_shape = c(24, 2048, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(1, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 24, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')

summary(cnn_model)

cnn_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

cnn_history <- cnn_model %>% fit(
  aaa_res, to_categorical(as.numeric(class)-1, 3),
  batch_size = 2000,
  epochs = 10,
  validation_split = 0.2
)


table(max.col(predict(cnn_model, aaa_test)))

# We're almost there... To (try to) explain which features are the best we've used an old well-known friend: LOCO! Taking one of the best result we've obtain, which data is formed by the quantities Energy, Entropy, Centroid frequency, Peak frequency, THD (total harmonic distortion), Crest factor, Spectral flatness, Spectral skewness, Spectral kurtosis calculated for each feature and the model is a very dumb multinomial, we've applied a LOCO using accuracy as metric and bootstrap to build confidence interval.
# Split train in two parts
train_indices <- createDataPartition(train$y, p = .75, list = FALSE)
D1 <- train[train_indices,6:ncol(train)]
D2 <- train[-train_indices,6:ncol(train)]

fun_dentro <- function(data, ind, feat, model_true, model_j){
  boot_data <- data[ind,]
  true <- predict(model_true, 
                  as.matrix(boot_data[,-217]),
                  type = "class")
  meno_j <- predict(model_j, 
                    as.matrix(boot_data[,-c(feat,217)]), 
                    type = "class")
  return(sum(abs(as.numeric(as.matrix(boot_data[,217]))-as.numeric(meno_j)))/sum(abs(as.numeric(as.matrix(boot_data[,217]))-as.numeric(true))))
}

model_true <- multinom(y ~ ., D1, maxit = 2500, trace = FALSE)

mod_for <- list()
# train a lot of models, removing a feature at time
for (i in 1:216){
  print(i)
  mod_for[[i]] <- multinom(y ~ ., D1[,-i], maxit = 2500, trace = FALSE)
}

b <- list()
b_median <- rep(NA,216)

# build confidence intervals
for (f in 1:216){
  b[[f]] <- boot(D2,
                 statistic = fun_dentro,
                 R = 400,
                 feat = f,
                 model_true = model_true,
                 model_j = mod_for[[f]], 
                 parallel = "multicore")
  b_median[f] <- quantile(b[[f]]$t, probs=c(.05,.5,.95))
  print(b_median[f])
}

LOCO <- which(b_median[,1]>0)

# Could be useful taking only the features whose interval doesn't contain the zero. They are  58
m <- multinom(y ~ ., new_data_[,c(LOCO,217)])

predictions <- predict(m,new_test_)
predictions

# This plot represent how many time a column come from by a particolar feature: it seems that the first, the eighth, the sixth and the thirteenth features are the most represented.

hist(as.integer(LOCO/9),breaks=24, main="Best Features", xlab="Features")

# By this plot we could say the statistic energy is more considerated.

hist(LOCO %% 9, breaks=9, main="Statistics", xlab="Features")


# Ensemble
# Finally, after all the models we tried, THE IDEA!!! Let's mix togheter the results from 4 of the best models that we had!
# 
# The result? Pure magic (at least we hope)...
# 
# This is the magical algorithm that we "invented" to ensemble the results.

sub15   <- read.csv("sub15.csv")
sub16   <- read.csv("sub16.csv")
sub29   <- read.csv("sub29.csv")
NNpower <- read.csv("NNpower.csv")

final <- data.frame(cbind(sub15=sub15[,2], sub16=sub16[,2], 
                          sub29=sub29[,2], NNpower=NNpower[,2])
                    
                    final$y <- rep(NA, nrow(final))
                    
                    for (i in 1:nrow(final)){
                      
                      row <- as.numeric(final[i, 1:4])
                      
                      if (sum(row == 3) > 2){
                        final[i, 7] <- 3
                      } 
                      
                      else if (row[4]==2) {
                        final[i, 7] <- 2
                      }
                      
                      else if (sum(row == 1) > 2) {
                        final[i, 7] <- 1
                      }
                      
                      else if (sum(row == 2) > 2) {
                        final[i, 7] <- 2
                      }
                      
                      else{
                        final[i, 7] <- round(row[1]*0.4 + row[2]*0.4 + row[3]*0.1 + row[4]*0.1)
                      }
                      
                    }
                    
                    table(final)
                    
submission <- data.frame(id=test$id, y=as.numeric(final))
write.csv(submission, file = "Nnpower_v4.csv", row.names = FALSE)