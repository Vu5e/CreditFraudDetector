# Credit Fraud Detector

## Introduction
Predictive models will be tested in this study for their accuracy in distinguishing between legitimate payments and fraudulent ones. In order to protect user privacy, the feature names and scales are not displayed in the dataset. However, we may still examine some of the dataset's most essential elements. Start now!

## Goals
- Understand the little distribution of the "little" data that was provided to us.
- Create a 50/50 sub-dataframe ratio of "Fraud" and "Non-Fraud" transactions.
- Determine the Classifiers we are going to use and decide which one has a higher accuracy.
- Create a Neural Network and compare the accuracy to our best classifier.

## Summary
- The transaction amount is relatively small. The mean of all the mounts made is approximately USD 88.
- There are no "Null" values, so we don't have to work on ways to replace values.
- Most of the transactions were Non-Fraud (99.83%) of the time, while Fraud transactions occurs (017%) of the time in the dataframe.

![Figure 1](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%201.png)

In the beginning, our dataset was extremely skewed. There isn't much fraud going on here. Our algorithms will undoubtedly overfit if we utilise this dataframe as the basis for our prediction models and analyses because it "assumes" that most transactions are not fraudulent. Instead of assuming, we need a system that can recognise patterns that indicate fraud!

![Figure 2](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%202.png)

We can get a sense of how skewed these characteristics are by looking at the distributions, and we can also see how the other features are distributed.

![Figure 3](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%203.png)

### Splitting the Data

Before proceeding with the Random UnderSampling technique we have to separate the orginal dataframe. Why? for testing purposes, remember although we are splitting the data when implementing Random UnderSampling or OverSampling techniques, we want to test our models on the original testing set not on the testing set created by either of these techniques. The main goal is to fit the model either with the dataframes that were undersample and oversample (in order for our models to detect the patterns), and test it on the original testing set.

![Figure 4](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%204.png)

### Random Undersampling
In this phase of the project we will implement "Random Under Sampling" which basically consists of removing data in order to have a more balanced dataset and thus avoiding our models to overfitting.

Steps:
- The first thing we have to do is determine how imbalanced is our class (use "value_counts()" on the class column to determine the amount for each label)
- Once we determine how many instances are considered fraud transactions (Fraud = "1") , we should bring the non-fraud transactions to the same amount as fraud transactions (assuming we want a 50/50 ratio), this will be equivalent to 492 cases of fraud and 492 cases of non-fraud transactions.
- After implementing this technique, we have a sub-sample of our dataframe with a 50/50 ratio with regards to our classes. Then the next step we will implement is to shuffle the data to see if our models can maintain a certain accuracy everytime we run this script.

Note: Our categorization models may not be as accurate as we'd like as a result of "Random Under-Sampling," because of the substantial data loss that it causes (bringing 492 non-fraud transaction from 284,315 non-fraud transaction)

![Figure 5](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%205.png)

### Equally Distributing and Correlating

Now that we have our dataframe correctly balanced, we can go further with our analysis and data preprocessing.
![Figure 6](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%206.png)

### Correlation Matrices
Correlation matrices are the essence of understanding our data. We want to know if there are features that influence heavily in whether a specific transaction is a fraud. However, it is important that we use the correct dataframe (subsample) in order for us to see which features have a high positive or negative correlation with regards to fraud transactions.

#### Summary and Explanation
- Negative Correlations: V17, V14, V12 and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.
- Positive Correlations: V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.
- BoxPlots: We will use boxplots to have a better understanding of the distribution of these features in fradulent and non fradulent transactions.

![Figure 7](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%207.png)

![Figure 8](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%208.png)

![Figure 9](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%209.png)

![Figure 10](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2010.png)

### Outlier Detection

Main aim in this section is to remove "extreme outliers" from features that have a high correlation with our classes. This will have a positive impact on the accuracy of our models.

#### Interquartile Range Method
- Interquartile Range (IQR): We calculate this by the difference between the 75th percentile and 25th percentile. Our aim is to create a threshold beyond the 75th and 25th percentile that in case some instance pass this threshold the instance will be deleted.
- Boxplots: Besides easily seeing the 25th and 75th percentiles (both end of the squares) it is also easy to see extreme outliers (points beyond the lower and higher extreme)

#### Outlier Removal Tradeoff:

We must exercise caution when deciding how high to set the bar for eliminating outliers. Multiplying a certain amount (ex: 1.5) by the threshold is how we arrive at the figure (Interquartile Range). The lower the threshold, the fewer outliers it will find, and the higher the threshold, the more outliers it will discover (ex: 3).

Tradeoff: For our purposes, we'd rather focus on "extreme" rather than "outliers," therefore a lower threshold would be preferable. Why? because if we do, we face the danger of losing data and thereby reducing the accuracy of our models. Our categorization models can be improved by adjusting this threshold.

### Summary

- Visualize Distributions: We first start by visualizing the distribution of the feature we are going to use to eliminate some of the outliers. V14 is the only feature that has a Gaussian distribution compared to features V12 and V10.
- Determining the threshold: After we decide which number we will use to multiply with the iqr (the lower more outliers removed), we will proceed in determining the upper and lower thresholds by substrating q25 - threshold (lower extreme threshold) and adding q75 + threshold (upper extreme threshold).
- Conditional Dropping: Lastly, we create a conditional dropping stating that if the "threshold" is exceeded in both extremes, the instances will be removed.
- Boxplot Representation: Visualize through the boxplot that the number of "extreme outliers" have been reduced to a considerable amount.

Note: Using outlier reduction, we've increased our accuracy by almost 3%! The accuracy of our models can be distorted by outliers, but we must be careful not to lose too much information or risk underfitting.

![Figure 11](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2011.png)

![Figure 12](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2012.png)

### Dimensionality Reduction and Clustering:

Understanding t-SNE: In order to understand this algorithm you have to understand the following terms:
- Euclidean Distance
- Conditional Probability
- Normal and T-Distribution Plots

Note: If you want a simple instructive video look at StatQuest: t-SNE, Clearly Explained by Joshua Starmer

Summary:
- t-SNE algorithm can pretty accurately cluster the cases that were fraud and non-fraud in our dataset.
- Although the subsample is pretty small, the t-SNE algorithm is able to detect clusters pretty accurately in every scenario (I shuffle the dataset before running t-SNE)
- This gives us an indication that further predictive models will perform pretty well in separating fraud cases from non-fraud cases.

![Figure 13](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2013.png)

![Figure 14](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2014.png)

### Classifiers (UnderSampling):

In this section we will train four types of classifiers and decide which classifier will be more effective in detecting fraud transactions. Before we have to split our data into training and testing sets and separate the features from the labels.

Summary:
- Logistic Regression classifier is more accurate than the other three classifiers in most cases. (We will further analyze Logistic Regression)
- GridSearchCV is used to determine the paremeters that gives the best predictive score for the classifiers.
- Logistic Regression has the best Receiving Operating Characteristic score (ROC), meaning that LogisticRegression pretty accurately separates fraud and non-fraud transactions.

Learning Curves:
- The wider the gap between the training score and the cross validation score, the more likely your model is overfitting (high variance).
- If the score is low in both training and cross-validation sets this is an indication that our model is underfitting (high bias)
- Logistic Regression Classifier shows the best score in both training and cross-validating sets.

![Figure 15](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2015.png)

![Figure 16](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2016.png)

![Figure 17](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2017.png)

![Figure 18](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2018.png)

![Figure 19](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2019.png)

![Figure 20](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2020.png)

Terms:
- True Positives: Correctly Classified Fraud Transactions
- False Positives: Incorrectly Classified Fraud Transactions
- True Negative: Correctly Classified Non-Fraud Transactions
- False Negative: Incorrectly Classified Non-Fraud Transactions
- Precision: True Positives/(True Positives + False Positives)
- Recall: True Positives/(True Positives + False Negatives)
- Precision as the name says, says how precise (how sure) is our model in detecting fraud transactions while recall is the amount of fraud cases our model is able to detect.
- Precision/Recall Tradeoff: The more precise (selective) our model is, the less cases it will detect. Example: Assuming that our model has a precision of 95%, Let's say there are only 5 fraud cases in which the model is 95% precise or more that these are fraud cases. Then let's say there are 5 more cases that our model considers 90% to be a fraud case, if we lower the precision there are more cases that our model will be able to detect.

Summary:
- Precision starts to descend between 0.90 and 0.92 nevertheless, our precision score is still pretty high and still we have a descent recall score.

![Figure 21](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2021.png)

![Figure 22](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2022.png)

![Figure 23](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2023.png)

![Figure 24](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2024.png)

SMOTE Technique
Understanding SMOTE:

- Solving the Class Imbalance: SMOTE creates synthetic points from the minority class in order to reach an equal balance between the minority and majority class.
- Location of the synthetic points: SMOTE picks the distance between the closest neighbors of the minority class, in between these distances it creates synthetic points.
- Final Effect: More information is retained since we didn't have to delete any rows unlike in random undersampling.
- Accuracy || Time Tradeoff: Although it is likely that SMOTE will be more accurate than random under-sampling, it will take more time to train since no rows are eliminated as previously stated.

Overfitting during Cross Validation:
In our undersample analysis I want to show you a common mistake I made that I want to share with all of you. It is simple, if you want to undersample or oversample your data you should not do it before cross validating. Why because you will be directly influencing the validation set before implementing cross-validation causing a "data leakage" problem. In the following section you will see amazing precision and recall scores but in reality our data is overfitting!

As mentioned previously, if we get the minority class ("Fraud) in our case, and create the synthetic points before cross validating we have a certain influence on the "validation set" of the cross validation process. Remember how cross validation works, let's assume we are splitting the data into 5 batches, 4/5 of the dataset will be the training set while 1/5 will be the validation set. The test set should not be touched! For that reason, we have to do the creation of synthetic datapoints "during" cross-validation and not before, just like below:

![Figure 25](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2025.png)

![Figure 26](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2026.png)

![Figure 27](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2027.png)

![Figure 28](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2028.png)

### Test Data with Logistic Regression:

Confusion Matrix:
Positive/Negative: Type of Class (label) ["No", "Yes"] True/False: Correctly or Incorrectly classified by the model.

True Negatives (Top-Left Square): This is the number of correctly classifications of the "No" (No Fraud Detected) class.

False Negatives (Top-Right Square): This is the number of incorrectly classifications of the "No"(No Fraud Detected) class.

False Positives (Bottom-Left Square): This is the number of incorrectly classifications of the "Yes" (Fraud Detected) class

True Positives (Bottom-Right Square): This is the number of correctly classifications of the "Yes" (Fraud Detected) class.

Summary:
- Random UnderSampling: We will evaluate the final performance of the classification models in the random undersampling subset. Keep in mind that this is not the data from the original dataframe.
- Classification Models: The models that performed the best were logistic regression and support vector classifier (SVM)

![Figure 29](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2029.png)

![Figure 30](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2030.png)

![Figure 31](https://github.com/Vu5e/CreditFraudDetector/blob/main/Images/Image%2031.png)

Conclusion:
We were able to correct our label imbalance by using SMOTE on our unbalanced dataset (more no fraud than fraud transactions). Many non-fraud transactions are mistakenly classified as fraud in our undersample data because our algorithm is unable to identify them appropriately. If customers who regularly made purchases had their cards suspended because our model determined that the transaction was fraudulent, the financial institution would be at a severe disadvantage. Customer complaints and customer dissatisfaction are expected to rise. After removing outliers from our oversample dataset, we will evaluate if our test set accuracy increases.
