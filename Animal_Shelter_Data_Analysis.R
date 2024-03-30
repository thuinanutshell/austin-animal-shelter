install.packages("janitor")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("tidymodels")
install.packages("glmnet")
install.packages("pROC")

library(dplyr)
library(readr)
library(janitor)
library(ggplot2)
library(tidymodels)
library(glmnet)
library(pROC)
aac_intakes_outcomes <- read_csv("Downloads/archive/aac_intakes_outcomes.csv")
View(aac_intakes_outcomes)

# DATA ClEANING
nrow(aac_intakes_outcomes)

# Remove NAs or empty entries (Intake Type)
missing_data <- is.na(aac_intakes_outcomes$`age_upon_intake_(years)` |
                is.na(aac_intakes_outcomes$intake_condition) |
                is.na(aac_intakes_outcomes$intake_type) |
                is.na(aac_intakes_outcomes$breed)
                      )
sum(missing_data)

# Remove all entries except for dogs
dog_data <- aac_intakes_outcomes[aac_intakes_outcomes$animal_type == "Dog",]
dog_data <- dog_data %>% clean_names()

# Remove rows with the Euthanasia Request intake type
cleaned_data <- dog_data[dog_data$intake_type != "Euthanasia Request", ]

# Create a new column feature
shelter_dog_data <- cleaned_data %>% mutate(intake_condition_new = case_when(
  intake_condition %in% c("Normal", "Injured", "Sick") ~ intake_condition,
  intake_condition %in% c("Nursing", "Pregnant") ~ "Maternity",
  TRUE ~ "Other"
))

# Convert intake_type column used for the model to a factor
shelter_dog_data$intake_type <- as.factor(shelter_dog_data$intake_type)
View(shelter_dog_data)

# LINEAR REGRESSION

# Model Summary
head(shelter_dog_data)
var_function <- time_in_shelter_days ~ intake_type + intake_condition_new + age_upon_intake_years
sum(shelter_dog_data$intake_type == "Owner Surrender")
sum(shelter_dog_data$intake_type == "Public Assist")
sum(shelter_dog_data$intake_type == "Stray")
sum(shelter_dog_data$intake_condition_new == "Injured")
sum(shelter_dog_data$intake_condition_new == "Normal")
sum(shelter_dog_data$intake_condition_new == "Sick")
sum(shelter_dog_data$intake_condition_new == "Maternity")
sum(shelter_dog_data$intake_condition_new == "Other")
lm1 <- lm(var_function, data = shelter_dog_data)
model.matrix(var_function, data = shelter_dog_data)
summary(lm1)  

# Confidence Interval Plotting
filtered_data <- shelter_dog_data[shelter_dog_data$intake_type == "Stray" & shelter_dog_data$intake_condition == "Normal", ]
model <- lm(time_in_shelter_days ~ age_upon_intake_years, data = filtered_data)
age_values <- c(0.1, 0.5, 1, 2, 3, 5, 7, 10, 12, 15, 18, 20)
result_df <- data.frame(age_upon_intake_years = numeric(0), mean_time_in_shelter_days = numeric(0), lower_ci = numeric(0), upper_ci = numeric(0))
for (age_value in age_values) {
  lm_model <- lm(time_in_shelter_days ~ age_upon_intake_years, data = filtered_data)
  pred <- predict(lm_model, newdata = data.frame(age_upon_intake_years = age_value), interval = "confidence", level = 0.95)
  result_df <- rbind(result_df, data.frame(age_upon_intake_years = age_value, 
                                           mean_time_in_shelter_days = pred[1], 
                                           lower_ci = pred[2], 
                                           upper_ci = pred[3]))
}
ggplot(result_df, aes(x = age_upon_intake_years, y = mean_time_in_shelter_days)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.2) +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), fill = "grey", alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +  # Add a linear fitting line
  labs(title = "95% Confidence Interval of Time in Shelter (Days)",
       x = "Age Upon Intake (Years)",
       y = "Mean Time in Shelter (Days)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", family = "Arial", size = 16),
    axis.text = element_text(family = "Arial", size = 11, margin = margin(t = 25, r = 25)),
    axis.title = element_text(family = "Arial", margin = margin(t = 25, r = 25)),
    axis.ticks = element_line(color = "black", size = 0.5),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    plot.margin = margin(t = 20, r = 20, b = 20, l = 20))

# CLASSIFICATION
non_live_release_outcomes <- c('Died', 'Disposal', 'Euthanasia', 'Missing')
shelter_dog_data$live_release <- ifelse(shelter_dog_data$outcome_type %in% non_live_release_outcomes, 0, 1)

# Additional feature (Sex & Breed)
shelter_dog_data$neutered_spayed_dogs <- ifelse(grepl("Neutered Male|Spayed Female", shelter_dog_data$sex_upon_intake), 1, 0)
shelter_dog_data$mix_breed <- ifelse(grepl("Mix", shelter_dog_data$breed), 1, 0)
View(shelter_dog_data)

# Split data into train and test
set.seed(130)
sample <- sample(c(TRUE, FALSE), nrow(shelter_dog_data), replace=TRUE, prob=c(0.8,0.2))
train_data <- shelter_dog_data[sample, ]
test_data <- shelter_dog_data[!sample, ]

# Logistic Regression on Training Set
logit_model <- glm(
  live_release ~ intake_type + intake_condition_new + age_upon_intake_years + neutered_spayed_dogs + mix_breed,
  data = train_data,
  family = "binomial"
)

summary(logit_model)

# Extract the coefficient, standard error, and z-value for age_upon_intake_years
age_coef <- coef(logit_model)["age_upon_intake_years"]
age_se <- summary(logit_model)$coefficients["age_upon_intake_years", "Std. Error"]
age_z <- qnorm(0.975)  # For a 95% confidence interval

# Calculate the confidence interval
age_ci <- age_coef + c(-1, 1) * age_z * age_se

# Print the coefficient and 95% confidence interval
cat("Coefficient for age_upon_intake_years:", age_coef, "\n")
cat("95% Confidence Interval:", age_ci, "\n")

# Predict live release probabilities
live_release_probs <- predict(logit_model, newdata = test_data, type = "response")

# Convert probabilities to binary outcomes using a threshold of 0.5
predicted_live_release <- ifelse(live_release_probs > 0.5, 1, 0)

# Create a confusion matrix
conf_matrix <- table(test_data$live_release, predicted_live_release)

# Print the confusion matrix
print(conf_matrix)

# Function to calculate confusion matrix for different thresholds
calculate_conf_matrix <- function(threshold) {
  predicted_live_release <- ifelse(live_release_probs > threshold, 1, 0)
  return(table(test_data$live_release, predicted_live_release))
}

# Trying different thresholds (e.g., 0.4, 0.6, 0.7) and evaluating performance
thresholds <- c(0.4, 0.6, 0.7)

for (thresh in thresholds) {
  conf_matrix <- calculate_conf_matrix(thresh)
  print(paste("Confusion Matrix with Threshold", thresh, ":"))
  print(conf_matrix)
}

# Predict live release probabilities on test data
live_release_probs <- predict(logit_model, newdata = test_data, type = "response")


# Predict live release probabilities
live_release_probs <- predict(logit_model, newdata = test_data, type = "response")

# Create a ROC curve
roc_curve <- roc(test_data$live_release, live_release_probs)

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)

# Add labels and legend
legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 2)), col = "blue", lwd = 2)


