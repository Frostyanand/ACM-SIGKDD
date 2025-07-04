#  Machine Learning Loss Functions – Deep Dive

##  Objective
This project presents a complete deep dive into the most commonly used **loss functions** in Machine Learning and Deep Learning, with visualizations, mathematical intuition, and code implementations.

> Loss functions are the heart of model training — they guide how the model learns. This notebook documents and demonstrates the purpose, behavior, and gradients of key loss functions, along with their practical usage.

---

##  Loss Functions Covered

### 1.  Mean Absolute Error (MAE)
- **Formula**:  
  \[
  \text{MAE} = \frac{1}{n} \sum |y - \hat{y}|
  \]
- **Intuition**: Measures the average magnitude of error — treats all errors equally.
- **Derivative**: Either -1, 0, or +1 (constant gradient)
- **Use case**: When you want a robust metric not sensitive to outliers.
- **Graph**: V-shaped loss curve

 _Insert MAE graph here_  
`![MAE Graph](images/mae_plot.png)`

---

### 2.  Mean Squared Error (MSE)
- **Formula**:  
  \[
  \text{MSE} = \frac{1}{n} \sum (y - \hat{y})^2
  \]
- **Intuition**: Penalizes large errors more than small ones (due to squaring).
- **Derivative**:  
  \[
  \frac{dL}{d\hat{y}} = 2(\hat{y} - y)
  \]
- **Use case**: Standard loss for regression tasks.
-  **Graph**: Smooth upward curve (parabola)

 _Insert MSE graph here_  
`![MSE Graph](images/mse_plot.png)`

---

### 3.  Binary Cross Entropy (BCE)
- **Formula**:  
  \[
  \text{BCE} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
  \]
- **Intuition**: Measures the distance between predicted probability and actual binary label.
- **Why use log?** It **penalizes confident wrong predictions** heavily.
- **Confidence**:  
  - Prediction near 0.5 → low confidence  
  - Prediction near 0 or 1 → high confidence  
- **Use case**: Binary classification with sigmoid output
- **Graph**: Steep log curve

 _Insert BCE graph here_  
`![BCE Graph](images/bce_plot.png)`

---

### 4.  Categorical Cross Entropy (CCE)
- **Formula**:  
  \[
  \text{CCE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
  \]
- **Labels**: Must be **one-hot encoded**
- **Prediction**: Output of **softmax** function (probability vector)
- **Use case**: Multi-class classification
- **Derivative (with softmax)**:  
  \[
  \frac{dL}{dz} = \hat{y} - y
  \]
- **Behavior**:  
  - Loss is low when correct class gets high probability  
  - Loss increases sharply when correct class gets low probability

 _Insert CCE prediction graph here_  
`![CCE Good vs Bad](images/cce_good_bad.png)`

---

### 5. ✳ Sparse Categorical Cross Entropy (Sparse CCE)
- Same logic as CCE, but:
  - True labels are **integers**, not one-hot vectors
- Useful for memory-efficient training in large-class tasks

 _Insert Sparse CCE example graph if available_  
`![Sparse CCE](images/sparse_cce.png)`

---

##  Softmax – What powers CCE
- Converts raw model scores into probabilities
- Formula:  
  \[
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}
  \]
- Ensures outputs sum to 1
- Makes loss function gradients interpretable

 _Insert Softmax plot here_  
`![Softmax](images/softmax_distribution.png)`

---

##  Comparison of Good vs Bad Predictions

###  Good Predictions
- Model gives high probability to correct class (e.g. 0.9)
- Loss is low

###  Bad Predictions
- Model gives low/confused probability to correct class (e.g. 0.25 or 0.4)
- Loss is much higher

 _Insert Good vs Bad graph_  
`![CCE Comparison](images/cce_comparison.png)`

---

##  Technologies Used

- Python 3.10+
- NumPy
- Matplotlib
- Jupyter/Colab Notebook

---

## 📁 File Structure

