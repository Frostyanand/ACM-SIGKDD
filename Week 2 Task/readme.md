#  Machine Learning Loss Functions

##  Objective
This project presents a complete deep dive into the most commonly used **loss functions** in Machine Learning and Deep Learning, with visualizations, mathematical intuition, and code implementations.

> Loss functions are the heart of model training — they guide how the model learns. This notebook documents and demonstrates the purpose, behavior, and gradients of key loss functions, along with their practical usage.
![image](https://github.com/user-attachments/assets/9998da36-629c-4dfc-ad1d-9d095aa85a0c)

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

![image](https://github.com/user-attachments/assets/f6a83e4e-ab57-4b4e-bfba-3203c1c804d3)

![image](https://github.com/user-attachments/assets/f0743075-0bfa-4a51-906c-02fd5633b655)

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
![image](https://github.com/user-attachments/assets/dab419d8-1e1e-490c-bd90-f7a335dbd69d)

![image](https://github.com/user-attachments/assets/32776051-3c93-4da7-b15e-51cabe3920b7)


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
![image](https://github.com/user-attachments/assets/b78e8aec-0da1-4972-ae7f-5ddcadc90b37)

![image](https://github.com/user-attachments/assets/4a7df362-dea5-4478-a310-2b1d736b2464)

![image](https://github.com/user-attachments/assets/09579082-7788-448e-926a-643e338ba225)

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

![image](https://github.com/user-attachments/assets/5fe9e8d0-2eb5-42be-9dc8-116d0bdc8bf1)


---

### 5. Sparse Categorical Cross Entropy (Sparse CCE)
- Same logic as CCE, but:
  - True labels are **integers**, not one-hot vectors
- Useful for memory-efficient training in large-class tasks

 _Insert Sparse CCE example graph if available_  
`![Sparse CCE](images/sparse_cce.png)`
![image](https://github.com/user-attachments/assets/0e71212b-46f6-4c7e-95e9-de8d45725347)

---

##  Softmax – What powers CCE
- Converts raw model scores into probabilities
- Formula:  
  \[
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}
  \]
- Ensures outputs sum to 1
- Makes loss function gradients interpretable



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

