# ML_interview



Here’s your **Advanced Analytics and Machine Learning Rapid-Fire Sheet** — neatly formatted, interview-ready, and perfect for copy-paste.  
Each question appears **exactly as you provided** (no edits), followed by a **crisp, well-presented answer**.

---

# 1\. What is regression and explains types with example?

**Answer:**  
Regression is a statistical method to model the relationship between a dependent variable and one or more independent variables.  
**Types:**

-   **Linear Regression:** Predicts continuous values (e.g., predicting house prices).
    
-   **Logistic Regression:** Used for binary outcomes (e.g., spam or not spam).
    
-   **Polynomial Regression:** Fits non-linear data.
    
-   **Ridge/Lasso Regression:** Adds penalty to reduce overfitting.
    

---

# 2\. What is meaning by samping in statistics? What different methods of sampling?

**Answer:**  
Sampling is selecting a subset of data from a population to make inferences.  
**Types:**

-   **Random Sampling**
    
-   **Stratified Sampling**
    
-   **Systematic Sampling**
    
-   **Cluster Sampling**
    
-   **Convenience Sampling**
    

---

# 3\. What do you understand by the term Normal distribution?

**Answer:**  
It’s a symmetric, bell-shaped curve where most data points cluster around the mean.  
**Properties:** Mean = Median = Mode, total area = 1.  
**Example:** Height or exam scores.

---

# 4\. What do you mean by categorical variable and which regression technique is used for analysis?

**Answer:**  
Categorical variables are non-numeric values representing categories (e.g., gender, color).  
**Technique:** Logistic Regression (for binary/multiclass outcomes).

---

# 5\. How do you decide whether your linear regression model fits the data?

**Answer:**  
By checking:

-   **R² / Adjusted R²** → Goodness of fit.
    
-   **Residual plots** → Randomly scattered = good fit.
    
-   **P-values** → < 0.05 = significant predictors.
    
-   **F-statistic** → Overall model significance.
    

---

# 6\. How can you overcome over-fitting?

**Answer:**

-   Use **regularization (L1/L2)**.
    
-   **Cross-validation**.
    
-   **Simplify model / reduce features**.
    
-   **Early stopping**.
    
-   **Add dropout (for deep learning)**.
    

---

# 7\. Is it possible to perform logistics regression with Microsoft Excel?

**Answer:**  
Yes, using **Excel’s Analysis ToolPak**, Solver add-in, or by fitting a logistic model using formulas.

---

# 8\. What do you understand by hypothesis? Explain the significance of P-value in hypothesis.

**Answer:**  
Hypothesis is an assumption about population parameters.  
**P-value** shows the probability of observing the result if the null hypothesis is true.  
If **P < α (e.g., 0.05)** → reject null hypothesis → result is statistically significant.

---

# 9\. What is z score in statistics?

**Answer:**  
Z-score measures how many standard deviations a data point is from the mean.  
**Formula:** (X - μ) / σ  
Used for standardization and outlier detection.

---

# 10\. What is F statistics?

**Answer:**  
Used to compare variances or test overall significance in regression.  
If **F > critical value**, model is significant.

---

# 11\. Give an example of inferential statistics?

**Answer:**  
Using a random sample to estimate the average height of all students in a college.  
Other examples: hypothesis testing, confidence intervals.

---

# 12\. What is significance of probability in statistics?

**Answer:**  
Probability quantifies uncertainty — used to make predictions, test hypotheses, and measure likelihood of outcomes.

---

# 13\. What is null hypothesis? Explain with example.

**Answer:**  
It states there is **no significant effect or difference**.  
**Example:** H₀: Students using new teaching method score same as traditional method.

---

# 14\. What is ARIMA model? Where it is used?

**Answer:**  
**ARIMA (Auto-Regressive Integrated Moving Average)** is used for **time series forecasting**.  
It models autocorrelations between observations.  
**Used in:** stock prediction, demand forecasting, temperature trends.

---

# 15\. How will you perform EDA? Which packages and functions you will use?

**Answer:**  
**Steps:**

-   Check nulls, data types, summary stats, correlations, outliers, distributions.  
    **Packages:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`.  
    **Functions:** `.info()`, `.describe()`, `.isnull()`, `corr()`, `histplot()`, `boxplot()`.
    

---

# 16\. What is Supervised and Unsupervised machine learning?

**Answer:**

-   **Supervised:** Uses labeled data (Regression, Classification).
    
-   **Unsupervised:** No labels; finds patterns (Clustering, Dimensionality Reduction).
    

---

# 17\. What is a training data set, cross validation test set and test dataset in machine learning? Explain.

**Answer:**

-   **Training set:** Used to train model.
    
-   **Validation set:** Used to tune parameters.
    
-   **Test set:** Used to evaluate final model performance.
    

---

# 18\. How much data will you allocate for training, validation and test sets? And why?

**Answer:**  
Typical split: **70%-15%-15%** or **80%-10%-10%**.  
Training needs most data to learn, others ensure generalization.

---

# 19\. You are given a data set. The data set has missing values which spread along 1 standard deviation from the median. What percentage of data would remain unaffected? Why?

**Answer:**  
About **68%** of data lies within ±1 standard deviation → thus, **68% remains unaffected**.

---

# 20\. While working on a dataset, how do you select important variables?

**Answer:**

-   **Correlation matrix**
    
-   **Feature importance (Tree models)**
    
-   **Recursive Feature Elimination (RFE)**
    
-   **P-values**
    
-   **Regularization (Lasso/Ridge)**
    

---

# 21\. What is the difference between Type 1 and Type 2 error?

**Answer:**

-   **Type I:** Rejecting true null hypothesis (false positive).
    
-   **Type II:** Accepting false null hypothesis (false negative).
    

---

# 22\. What is over-fitting and under-fitting? How do you ensure you're not overfitting with a model?

**Answer:**

-   **Overfitting:** Fits training data too well, poor generalization.
    
-   **Underfitting:** Too simple, poor on both training and test data.  
    **Prevention:** Cross-validation, regularization, simpler model, dropout.
    

---

# 23\. How do you handle missing and corrupted data in a dataset?

**Answer:**

-   **Remove** rows/columns (if small % missing).
    
-   **Impute** using mean/median/mode.
    
-   **Model-based imputation.**
    
-   **Use interpolation** for time series.
    

---

# 24\. How do you think Google is training data for self-driving cars?

**Answer:**  
Using **millions of sensor, camera, and LIDAR data points** labeled with real-world driving scenarios; trained with **deep learning (CNNs + Reinforcement Learning)**.

---

# 25\. How will you know which machine learning algorithm to choose for your classification problem?

**Answer:**  
Depends on:

-   Data size and type
    
-   Linearity (Linear vs Non-linear)
    
-   Accuracy vs interpretability  
    **Examples:**  
    Logistic Regression → simple, interpretable  
    Random Forest → high accuracy  
    SVM → small datasets with clear margins
    

---

# 26\. What type of data does the machine learning model handle - categorical, etc..?

**Answer:**  
ML models handle **categorical, numerical, ordinal, textual, image, audio** data.  
Categorical is encoded via **LabelEncoder/OneHotEncoder**.

---

# 27\. What is your favorite machine learning algorithm and why?

**Answer:**  
**Random Forest:** Robust, handles missing values, reduces overfitting, works well for classification & regression with minimal tuning.

---

# 28\. You are given a data set consisting of variables having more than 30% missing values? Let's say, out of 50 variables, B variables have missing values higher than 30%. How will deal with them?

**Answer:**

-   Drop variables with >30% missing (if not critical).
    
-   Impute using **advanced methods (KNN Imputer, MICE)** if important.
    
-   Analyze cause of missingness before removing.
    

---

# 29\. Which Python libraries are used for machine learning? How you can save a trained model and load it back?

**Answer:**  
**Libraries:** `scikit-learn`, `tensorflow`, `keras`, `pandas`, `numpy`, `xgboost`.  
**Save/Load model:**

```python
import joblib
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
```

---

# 30\. What is the difference between Artificial intelligence, machine leaming, deep learning and natural network processing?

**Answer:**

-   **AI:** Simulating human intelligence.
    
-   **ML:** Learning patterns from data.
    
-   **DL:** ML using neural networks.
    
-   **NLP:** AI branch for text/speech understanding.
    

---

# 31\. What is decision tree?

**Answer:**  
A tree-like model for decisions.  
Nodes = conditions, Leaves = outcomes.  
Used for both regression and classification.

---

# 32\. What is F1-score? Why it is better than accuracy?

**Answer:**  
F1 = Harmonic mean of Precision & Recall.  
Better than accuracy for **imbalanced datasets** as it considers both false positives & negatives.

---

# 33\. What is classification? Different metrics you use in classification?

**Answer:**  
Classification predicts discrete categories.  
**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.

---

# 34\. What is difference between classification and regression?

**Answer:**

-   **Classification:** Predicts category (spam/not spam).
    
-   **Regression:** Predicts continuous values (price, salary).
    

---

# 35\. What is Generative AI?

**Answer:**  
AI that generates new content (text, image, music) from learned data patterns.  
**Examples:** ChatGPT, DALL·E, Midjourney.

---

# 36\. What is a Generative Adversarial Network (GAN)?

**Answer:**  
GAN consists of two neural networks — **Generator** and **Discriminator** — competing with each other to create realistic data.

---

# 37\. Explain the concept of the Generator and Discriminator in GANs.

**Answer:**

-   **Generator:** Creates fake data to fool the Discriminator.
    
-   **Discriminator:** Distinguishes between real and fake data.  
    They improve each other through competition until Generator produces realistic outputs.
    

---

✅ **Tip:** Revise keywords (R², P-value, Overfitting, F1-Score, Regularization, ARIMA) — these are *favorite interview hotspots*.

---

Would you like me to make this into a **color-coded flashcard format (Q on front, short key points on back)** in a PDF for easy print or phone study?
