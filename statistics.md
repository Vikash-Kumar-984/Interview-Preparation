Comprehensive Statistics Q&A
T-Test and Student’s T-Distribution
1. What is a T-test, and when should it be used instead of a Z-test?
A T-test is a statistical hypothesis test used to determine if there is a significant difference between the means of two groups. You should use a T-test instead of a Z-test when the sample size is small (typically n\<30) and the population standard deviation (
sigma) is unknown. The T-test uses the sample standard deviation (s) as an estimate for 
sigma, which accounts for the extra uncertainty present in small samples.
<br>
[Data Science, Data Analyst], [Google, Flipkart], 3

2. Explain how a paired T-test differs from an independent T-test.
The key difference lies in the nature of the samples being compared:

An independent T-test (or two-sample T-test) compares the means of two separate, unrelated groups. For example, comparing the average test scores of students from two different schools.

A paired T-test compares the means of the same group at two different times or under two different conditions. It's used for "before and after" scenarios. For example, measuring the change in blood pressure for the same group of patients before and after taking a medication.

<br>
[Data Scientist, Data Engineer], [Amazon, Paytm], 2

3. What assumptions need to be met for a valid T-test?
For a T-test to be valid, several assumptions must be met:

Independence: The observations within each sample must be independent.

Normality: The data in each group should be approximately normally distributed.

Homogeneity of Variances (for independent T-tests): The variances of the two groups being compared should be roughly equal.

4. How would you implement a T-test in Python using scipy.stats?
You can use ttest_ind for an independent T-test and ttest_rel for a paired T-test.

Example: Independent T-test

from scipy import stats
import numpy as np

# Sample data for two independent groups
group1_scores = np.random.normal(loc=85, scale=5, size=25)
group2_scores = np.random.normal(loc=80, scale=6, size=25)

# Perform the independent T-test
t_statistic, p_value = stats.ttest_ind(group1_scores, group2_scores)

print(f"Independent T-test Results:")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

Example: Paired T-test

# Sample data for paired observations (e.g., before and after)
scores_before = np.random.normal(loc=75, scale=8, size=30)
scores_after = scores_before + np.random.normal(loc=5, scale=3, size=30)

# Perform the paired T-test
t_statistic_rel, p_value_rel = stats.ttest_rel(scores_before, scores_after)

print(f"\nPaired T-test Results:")
print(f"T-statistic: {t_statistic_rel:.4f}")
print(f"P-value: {p_value_rel:.4f}")

[Data Science, Business Analyst], [Microsoft, Swiggy], 3

5. What is Student's T-distribution, and how does it differ from the standard normal distribution?
Student's T-distribution is a probability distribution used to estimate population parameters when the sample size is small and/or the population standard deviation is unknown.

Key differences from the standard normal distribution:

Heavier Tails: The T-distribution has fatter tails, assigning higher probability to extreme values to account for uncertainty in small samples.

Shape depends on Degrees of Freedom: The shape is determined by degrees of freedom (df). As df increases, the T-distribution approaches the normal distribution.

Variance: The variance of the T-distribution is greater than 1, while the variance of the standard normal distribution is exactly 1.

<br>
[Data Science, Data Analyst], [Flipkart, Google], 3

6. When would you use Student's T-distribution over the normal distribution?
You would use the Student's T-distribution when:

The population standard deviation (
sigma) is unknown and you must use the sample standard deviation (s) as an estimate.

The sample size is small (typically n\<30).

<br>
[Data Scientist, Machine Learning Engineer], [Amazon, Zomato], 2

7. What are the properties of Student's T-distribution?
Symmetry: It is symmetric about its mean of 0.

Bell-Shaped: It has a bell shape, but it's shorter and wider than the normal distribution.

Degrees of Freedom (df): Its shape depends on the degrees of freedom. Lower df results in heavier tails.

Mean and Variance: The mean is 0 (for df1). The variance is df/(df−2) (for df2).

8. How do you calculate the degrees of freedom in a T-test using Student's T-distribution?
One-Sample T-test: df=n−1

Paired T-test: df=n−1 (where n is the number of pairs)

Independent T-test (equal variances): df=n_1+n_2−2

<br>
[Data Science, Business Analyst], [Microsoft, Swiggy], 3

9. What are the key differences between a T-test and a Z-test?
Feature

T-test

Z-test

Population Std Dev (
sigma)

Unknown (uses sample std dev, s)

Known

Sample Size (n)

Typically small (n\<30)

Typically large (n
ge30)

Underlying Distribution

Student's T-distribution

Standard Normal Distribution

<br>





[Data Science, Data Analyst], [Flipkart, Amazon], 3





10. When would you choose to use a T-test over a Z-test in a research scenario?
You would choose a T-test in a research scenario when you are working with a small sample and the standard deviation of the entire population is unknown. This is the most common situation in practical research, as population parameters are rarely known.
<br>
[Business Analyst, Data Scientist], [Google, Paytm], 2

11. Explain how sample size impacts the decision to use a T-test or Z-test.
Small Sample Size (n\<30): The sample standard deviation (s) is a less reliable estimate of the population standard deviation (
sigma). The T-distribution's heavier tails account for this uncertainty, so a T-test is required.

Large Sample Size (n
ge30): The sample standard deviation (s) becomes a very good estimate of 
sigma. The T-distribution becomes almost identical to the Z-distribution, so a Z-test can be used as a close approximation (though a T-test is still technically correct if 
sigma is unknown).

12. How does the assumption of population variance affect the choice between T-test and Z-test?
This is the most critical factor:

If the population variance (
sigma 
2
 ) is known, you should always use a Z-test.

If the population variance (
sigma 
2
 ) is unknown, you must estimate it from the sample, and you should always use a T-test.

Confidence Interval and Margin of Error
13. What is a confidence interval, and why is it important in inferential statistics?
A confidence interval (CI) is a range of values, derived from sample data, that is likely to contain the true value of an unknown population parameter. It's important because it quantifies the uncertainty of an estimate, providing a range of plausible values rather than just a single point estimate.
<br>
[Data Science, Data Analyst], [Amazon, Flipkart], 3

14. Explain the relationship between confidence intervals and the margin of error.
A confidence interval is constructed by taking a point estimate and adding and subtracting a margin of error.
Formula: Confidence Interval = Point Estimate ± Margin of Error
The margin of error quantifies the width of the interval around the point estimate.

15. How do you interpret a 95% confidence interval in statistical analysis?
The correct interpretation is: "If we were to take many random samples and construct a 95% confidence interval for each, we would expect about 95% of those intervals to contain the true population parameter." It refers to the reliability of the method, not the probability of a single interval containing the true value.

16. How can you calculate confidence intervals in Python using scipy.stats?
You can use the .interval() method from a distribution object (like stats.t).

import numpy as np
from scipy import stats

# Generate sample data
data = np.random.normal(loc=100, scale=15, size=50)

# Calculate sample statistics
sample_mean = np.mean(data)
se = stats.sem(data) # Standard error of the mean
n = len(data)

# Calculate the 95% confidence interval for the mean
ci_95 = stats.t.interval(confidence_level=0.95, df=n-1, loc=sample_mean, scale=se)

print(f"95% Confidence Interval: {ci_95}")

Chi-Square Test & Chi-Square Distribution
17. What is the Chi-square test, and when is it used in statistical analysis?
The Chi-square (
chi 
2
 ) test is a non-parametric test used to analyze categorical data. It is used in two main scenarios:

Chi-square Test of Independence: To determine if there is a significant association between two categorical variables.

Chi-square Goodness-of-Fit Test: To determine if an observed frequency distribution fits a theoretical distribution.

<br>
[Data Science, Business Analyst], [Flipkart, Amazon], 3

18. Explain how to perform a Chi-square test of independence.
State Hypotheses: H_0: The variables are independent. H_a: The variables are dependent.

Create a Contingency Table of observed frequencies.

Calculate Expected Frequencies for each cell: E=
frac(textRowTotal)times(textColumnTotal)textGrandTotal.

Calculate the Chi-square Statistic: 
chi 
2
 =
sum
frac(O−E) 
2
 E.

Determine p-value by comparing the statistic to a 
chi 
2
  distribution with df=(
textrows−1)
times(
textcols−1).

Conclusion: If p-value < significance level, reject H_0.

<br>
[Data Analyst, Data Scientist], [Google, Swiggy], 2

19. How is the Chi-square distribution related to categorical data analysis?
The Chi-square distribution is the sampling distribution for the Chi-square test statistic. It allows us to determine the p-value, which is the probability of observing a discrepancy between observed and expected frequencies as large as we did, purely by chance.

20. Provide a Python implementation of a Chi-square test using scipy.stats.
Use the chi2_contingency function.

import numpy as np
from scipy.stats import chi2_contingency

# Contingency table: [Male, Female] vs [Chocolate, Vanilla, Strawberry]
observed = np.array([[40, 30, 10], 
                     [50, 60, 30]])

chi2_stat, p_val, dof, expected = chi2_contingency(observed)

print(f"Chi-square Statistic: {chi2_stat:.4f}")
print(f"P-value: {p_val:.4f}")

Bayes' Theorem
21. What is Bayes' theorem, and how is it applied in data science?
Bayes' theorem is a formula for updating probabilities based on new evidence.


P(A∣B)= 
P(B)
P(B∣A)⋅P(A)
​
 

In data science, it's the foundation for Naive Bayes classifiers, used for tasks like spam filtering and text classification. It's also central to the field of Bayesian inference.
<br>
[Data Science, Machine Learning Engineer], [Amazon, Flipkart], 3

22. Explain how Bayes' theorem is used in spam filtering or medical diagnostics.
Spam Filtering: It calculates the probability an email is spam given the words it contains (P(
textSpam∣
textWords)). It learns the prior probability of spam and the likelihood of certain words appearing in spam vs. non-spam emails from a training set.

Medical Diagnostics: It calculates the probability a patient has a disease given a positive test result (P(
textDisease∣
textPositiveTest)). This correctly incorporates the prevalence of the disease (the prior) and the test's accuracy (the likelihood).

<br>
[Data Scientist, Data Analyst], [Google, Paytm], 2

23. What is the difference between prior, likelihood, and posterior probabilities in Bayes' theorem?
Prior Probability P(A): The initial belief in a hypothesis before seeing evidence.

Likelihood P(B∣A): The probability of observing the evidence if the hypothesis is true.

Posterior Probability P(A∣B): The updated belief in the hypothesis after considering the evidence.

24. How would you implement Bayes' theorem in Python for a classification problem?
The easiest way is to use a Naive Bayes classifier from scikit-learn.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample Data
X_train = ['free money offer', 'your meeting is scheduled']
y_train = ['spam', 'not spam']
X_test = ['get your prize now']

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train and predict
model = MultinomialNB()
model.fit(X_train_vec, y_train)
prediction = model.predict(X_test_vec)

print(f"Prediction: {prediction[0]}")

[Data Science, Data Engineer], [Microsoft, Swiggy], 3

Goodness of Fit Test
25. What is the goodness-of-fit test, and how is it used in statistical analysis?
A goodness-of-fit test determines how well an observed sample distribution fits a hypothesized theoretical distribution. It's used to check, for example, if a die is fair (fits a uniform distribution) or if data follows a normal distribution.
<br>
[Data Science, Business Analyst], [Google, Flipkart], 3

26. How is the Chi-square goodness-of-fit test performed?
State Hypotheses: H_0: The sample data fits the expected distribution. H_a: The data does not fit.

Collect Observed Frequencies (O) for each category.

Determine Expected Frequencies (E) based on the hypothesized distribution.

Calculate the Chi-square Statistic: 
chi 
2
 =
sum
frac(O−E) 
2
 E.

Determine p-value using a 
chi 
2
  distribution with df=k−1 (where k is the number of categories).

Conclusion: If p-value is low, reject H_0.

<br>
[Data Analyst, Data Scientist], [Amazon, Zomato], 2

27. When would you use a goodness-of-fit test for a dataset?
You use it when you have a single categorical variable and want to see if its frequency distribution matches a known or theoretical pattern. For example, testing if the distribution of M&M colors in a bag matches the company's stated percentages.

28. How can you implement a goodness-of-fit test in Python using scipy.stats?
Use the chisquare function.

from scipy.stats import chisquare

# Observed die rolls (120 total)
observed_frequencies = [25, 15, 22, 18, 20, 20]
# Expected for a fair die (120/6 = 20 each)
expected_frequencies = [20, 20, 20, 20, 20, 20]

chi2_stat, p_val = chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

print(f"Chi-square Statistic: {chi2_stat:.4f}")
print(f"P-value: {p_val:.4f}")

F-Distribution and F-Test
29. What is the F-distribution, and how is it used in statistical analysis?
The F-distribution is a probability distribution used in hypothesis testing. It is the ratio of two independent Chi-square variables. It is used primarily in:

Analysis of Variance (ANOVA): To test the equality of means across two or more groups.

Regression Analysis: To test the overall significance of a model.

<br>
[Data Science, Business Analyst], [Amazon, Flipkart], 3

30. Explain the relationship between the F-distribution and analysis of variance (ANOVA).
In ANOVA, the F-statistic is calculated as the ratio of the variance between groups to the variance within groups.


F= 
Variance within groups
Variance between groups
​
 

This F-statistic follows an F-distribution. By comparing our calculated F-statistic to the F-distribution, we can find the p-value to determine if there's a significant difference between the group means.
<br>
[Data Analyst, Data Scientist], [Google, Paytm], 2

31. What are the properties of the F-distribution in hypothesis testing?
Positively Skewed: It is always right-skewed and cannot be negative.

Defined by Two Degrees of Freedom: Its shape depends on numerator degrees of freedom (df_1) and denominator degrees of freedom (df_2).

Range: Values range from 0 to infinity.

32. How do you calculate the F-statistic in Python using scipy.stats?
You typically get it as an output from a function like stats.f_oneway for ANOVA.

from scipy import stats

group_a = [85, 86, 88, 75, 78, 94, 98, 79, 71, 80]
group_b = [91, 92, 93, 85, 86, 87, 94, 96, 82, 85]
group_c = [79, 78, 88, 94, 92, 85, 83, 85, 82, 81]

f_statistic, p_value = stats.f_oneway(group_a, group_b, group_c)

print(f"Calculated F-statistic: {f_statistic:.4f}")

33. What is an F-test, and how does it differ from other statistical tests?
An F-test is any test where the test statistic has an F-distribution. It differs from T-tests (comparing 1-2 means) and Chi-square tests (categorical data) because it is primarily used to compare variances. This application allows it to compare the means of 3+ groups (ANOVA) or test entire regression models.
<br>
[Data Science, Business Analyst], [Flipkart, Amazon], 3

34. Explain how an F-test is used in testing for the equality of variances.
An F-test can directly compare the variances of two populations. The F-statistic is the ratio of the two sample variances:


F= 
s 
2
2
​
 
s 
1
2
​
 
​
 

If the variances are equal, this ratio should be close to 1. A large F-statistic suggests the variances are different. The p-value is found by comparing this statistic to an F-distribution.
<br>
[Data Analyst, Data Scientist], [Google, Swiggy], 2

35. What assumptions must be met to perform an F-test?
Independence: The samples must be independent.

Normality: The populations from which the samples are drawn should be normally distributed. F-tests are sensitive to violations of this assumption.

36. How do you perform an F-test in Python using scipy.stats?
The most common F-test is part of ANOVA (stats.f_oneway). To compare two variances directly, you can calculate the F-statistic and use scipy.stats.f.sf to get the p-value.

import numpy as np
from scipy.stats import f

sample1 = np.random.normal(loc=10, scale=2, size=20)
sample2 = np.random.normal(loc=10, scale=3, size=25)

f_stat = np.var(sample2, ddof=1) / np.var(sample1, ddof=1)
df1 = len(sample2) - 1
df2 = len(sample1) - 1

p_value = f.sf(f_stat, df1, df2) * 2 # Two-tailed test

print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

[Data Science, Data Engineer], [Microsoft, Ola], 3

ANOVA and its Assumptions
37. What is ANOVA, and why is it used in statistical analysis?
ANOVA (Analysis of Variance) is a statistical method used to test for significant differences between the means of two or more groups. It is used instead of multiple T-tests to avoid inflating the Type I error rate (false positives). It works by comparing the variation between groups to the variation within groups.

38. Explain the assumptions that need to be met before conducting an ANOVA test.
Independence: Observations in each group are independent.

Normality: Data in each group is approximately normally distributed.

Homogeneity of Variances (Homoscedasticity): The variance within each group is approximately equal.

39. How does one-way ANOVA differ from two-way ANOVA?
One-Way ANOVA: Involves one categorical independent variable (factor). It tests if there is a difference in the dependent variable's mean across the levels of that single factor.

Two-Way ANOVA: Involves two categorical independent variables. It tests the main effect of each factor and also the interaction effect between them.

40. Provide a Python implementation of ANOVA using statsmodels or scipy.stats.
Using scipy.stats (One-Way):

from scipy import stats
group1 = [85, 86, 88, 75, 78]
group2 = [91, 92, 93, 85, 86]
group3 = [79, 78, 88, 94, 92]
f_stat, p_val = stats.f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat:.4f}, P-value: {p_val:.4f}")

Using statsmodels (More flexible, e.g., Two-Way):

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = {'yield': [20, 22, 25, 27, 15, 17, 21, 23],
        'fertilizer': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
        'soil': ['Clay', 'Clay', 'Clay', 'Clay', 'Loam', 'Loam', 'Loam', 'Loam']}
df = pd.DataFrame(data)

model = ols('yield ~ C(fertilizer) + C(soil) + C(fertilizer):C(soil)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

41. What are the different types of ANOVA, and when should each be used?
One-Way ANOVA: One categorical independent variable.

Two-Way ANOVA: Two categorical independent variables.

Repeated Measures ANOVA: For measuring the same subjects multiple times (longitudinal data).

MANOVA (Multivariate Analysis of Variance): When you have more than one continuous dependent variable.

<br>
[Data Science, Business Analyst], [Flipkart, Google], 3

42. Explain the difference between one-way ANOVA and two-way ANOVA.
The primary difference is the number of independent variables (factors).

One-Way ANOVA tests the effect of a single factor.

Two-Way ANOVA tests the effects of two factors simultaneously, including their potential interaction effect.

<br>
[Data Analyst]
