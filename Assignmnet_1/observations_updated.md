# Observations — Model Benchmark and Experiment Analysis

## 1. Overview

I evaluated several classification models on the Customer Churn dataset using a consistent train/validation/test setup. The main purpose of the experiments was not only to find the model with the highest score, but also to understand how different algorithms behave under different assumptions and configurations.

The experiments included Logistic Regression, KNN with two different values of K, shallow and deep Decision Trees, Random Forest, Gradient Boosting, and two SVM configurations. The notebook used stratified splitting into approximately 70% training, 15% validation, and 15% test data so that the class distribution was preserved across the splits.

The models were compared using accuracy, precision, recall, F1-score, ROC-AUC, training time, and inference time. This makes the comparison more meaningful than selecting a model from accuracy alone.

---

## 2. Why these models were selected

I used a range of models because each one represents a different type of learning approach.

- **Logistic Regression** was included as a simple and interpretable baseline. It provides a useful reference point for comparing more complex models.
- **KNN** was tested with `K=3` and `K=15` to observe the effect of the number of neighbors. A smaller K should make the model more sensitive to individual training examples, while a larger K should produce smoother and more generalized predictions.
- **Decision Tree** was tested in both shallow and deep forms. The shallow tree (`max_depth=3`) was intended to represent an underfitting/simple model, while the deep tree (`max_depth=20`) was included to observe the effect of a much more flexible model and possible overfitting.
- **Random Forest** was used to compare a single Decision Tree with an ensemble of trees. The expectation was that combining multiple trees could improve generalization.
- **Gradient Boosting** was included as a boosting-based ensemble method to determine whether sequentially correcting previous errors could improve performance further.
- **SVM** was tested in scaled and unscaled versions because SVM is sensitive to the scale of numerical features. This experiment directly tests whether appropriate feature scaling improves performance.

This selection allowed me to compare simple linear methods, distance-based methods, tree-based methods, ensemble methods, and margin-based methods.

---

## 3. Data preparation and evaluation setup

The notebook used stratified train/validation/test splitting. This was important because it keeps the class proportions more consistent between the datasets.

For the standard preprocessing pipeline:

- Numerical features were standardized using `Min Max Scaler`. Beacuse there were no outliers present in the data and I wanted to keep the uniform distriution preserved. 
- Categorical features were converted using `OneHotEncoder(handle_unknown='ignore')`.
- The preprocessing was fitted on the training data and then applied to validation and test data.

A separate unscaled preprocessing variant was used specifically for the SVM scaling experiment. This allowed the scaled and unscaled SVM configurations to be compared while keeping the rest of the experiment consistent.

The use of a validation set was useful for comparing models before considering their final test performance. The test set was kept separate for the final generalization comparison.

---

## 4. Main observations from the results

The results were much stronger overall than would be expected from a difficult classification problem. Most models achieved validation accuracy above 94%, and several models achieved validation ROC-AUC values above 0.99.

The strongest overall results came from the tree-based ensemble and deep-tree models:

- **Gradient Boosting** achieved validation accuracy of `0.998548`, validation F1 of `0.998720`, and ROC-AUC of `0.999988`.
- **Random Forest** achieved validation accuracy of `0.995766`, validation F1 of `0.996280`, and ROC-AUC of `0.999980`.
- **Deep Decision Tree** achieved validation accuracy of `0.999803`, validation F1 of `0.999827`, and ROC-AUC of `0.999810`.
- **Shallow Decision Tree** also performed very strongly, with validation accuracy of `0.992183` and recall of `1.000000`.
- **KNN with K=15** performed better than KNN with K=3 on validation accuracy and ROC-AUC, reaching `0.985908` accuracy and `0.999065` ROC-AUC.
- **Logistic Regression** produced a strong and stable baseline, with validation accuracy of `0.980011` and ROC-AUC of `0.996616`.
- **Scaled SVM** performed substantially better than the unscaled SVM, with validation accuracy of `0.978968` compared with `0.942709`.

The final test accuracy followed a similar pattern. The highest test accuracy in the supplied results was achieved by the deep Decision Tree at `0.999773`, followed by Gradient Boosting at `0.998488`, Random Forest at `0.995130`, and the shallow Decision Tree at `0.991577`.

---

## 5. Logistic Regression — baseline model

I used Logistic Regression as the baseline because it is relatively simple and gives an interpretable reference for the other models.

Its training accuracy was `0.980627`, validation accuracy was `0.980011`, and test accuracy was `0.980854`. The validation ROC-AUC was `0.996616`.

The small difference between training and validation accuracy suggests that the model generalized consistently rather than showing a large train-validation gap. This makes Logistic Regression a useful baseline even though it was not the highest-performing model.

The main reason I would keep Logistic Regression in the benchmark is that a more complex model should provide a meaningful improvement over a simple baseline. In this case, the ensemble and tree models clearly improved the measured performance.

---

## 6. KNN — effect of K

I compared two KNN configurations to study the effect of the number of neighbors.

### KNN with K=3

The K=3 model achieved:

- Training accuracy: `0.993771`
- Validation accuracy: `0.987858`
- Validation F1: `0.989261`
- Validation ROC-AUC: `0.995557`
- Test accuracy: `0.988416`

The relatively high training score compared with validation indicates that the smaller neighborhood is more sensitive to local training examples.

### KNN with K=15

The K=15 model achieved:

- Training accuracy: `0.988045`
- Validation accuracy: `0.985908`
- Validation F1: `0.987522`
- Validation ROC-AUC: `0.999065`
- Test accuracy: `0.986873`

Increasing K reduced training accuracy slightly, which is consistent with a smoother decision boundary. Interestingly, K=15 achieved a higher ROC-AUC than K=3.

Overall, both KNN configurations performed well, but KNN was much slower at inference than the other models. K=3 required about `33.67` seconds for inference, while K=15 required about `41.18` seconds. Therefore, although KNN produced strong predictive results, it was not the most efficient choice for inference.

---

## 7. Decision Tree — underfitting versus overfitting

The Decision Tree experiment was especially useful because I intentionally compared a shallow tree with a deep tree.

### Shallow Decision Tree

The shallow tree had `max_depth=3`. It achieved:

- Training accuracy: `0.991512`
- Validation accuracy: `0.992183`
- Validation recall: `1.000000`
- Validation F1: `0.993155`
- Validation ROC-AUC: `0.998219`
- Test accuracy: `0.991577`

The training and validation scores are very close. This suggests that the shallow model was not suffering from a large variance problem. However, limiting the tree depth also limits its ability to represent more complicated relationships.

For the purpose of the experiment, this model is useful because it shows what happens when model complexity is restricted.

### Deep Decision Tree

The deep tree used `max_depth=20`. It achieved:

- Training accuracy: `1.000000`
- Validation accuracy: `0.999803`
- Validation F1: `0.999827`
- Validation ROC-AUC: `0.999810`
- Test accuracy: `0.999773`

The model almost perfectly fitted the training data and also achieved extremely high validation and test performance. Therefore, the expected overfitting behavior is visible in the training result, but the supplied test results do not show a large drop in performance.

This is an important observation: a very high training score by itself is not enough to conclude that the model is unusable. The validation and test results must also be considered. In this experiment, the deep tree maintained very strong performance on the held-out data.

---

## 8. Random Forest — why use an ensemble?

Random Forest was included to see whether combining multiple decision trees could provide strong generalization while reducing the dependence on one individual tree.

The Random Forest achieved:

- Training accuracy: `0.995628`
- Validation accuracy: `0.995766`
- Validation precision: `0.993007`
- Validation recall: `0.999573`
- Validation F1: `0.996280`
- Validation ROC-AUC: `0.999980`
- Test accuracy: `0.995130`

The validation performance is extremely strong, and the train-validation scores are very close. This suggests that the ensemble generalized well in this experiment.

Random Forest also performed better than the shallow tree while avoiding the near-perfect training fit of the deep tree. This makes it a strong compromise between model flexibility and generalization.

However, training took approximately `27.36` seconds, which was considerably slower than Logistic Regression, KNN training, and the Decision Trees.

---

## 9. Gradient Boosting — strongest ensemble performance

Gradient Boosting was included to compare a boosting method against the Random Forest ensemble.

It produced some of the strongest results in the benchmark:

- Training accuracy: `0.998603`
- Validation accuracy: `0.998548`
- Validation precision: `0.998667`
- Validation recall: `0.998774`
- Validation F1: `0.998720`
- Validation ROC-AUC: `0.999988`
- Test accuracy: `0.998488`

The validation results are extremely strong and the training-validation gap is very small. This indicates that the model was able to learn a highly effective decision boundary without a major reduction in validation performance.

Compared with Random Forest, Gradient Boosting produced higher validation accuracy, F1, and ROC-AUC. It also produced a higher test accuracy.

The main disadvantage was computational cost. Training took approximately `105.90` seconds, which was by far the longest training time in the supplied results. Therefore, Gradient Boosting gave excellent predictive performance, but that performance came with a significantly higher training cost.

---

## 10. SVM — importance of feature scaling

The SVM experiment directly demonstrated why preprocessing matters for some algorithms.

### Scaled SVM

The scaled SVM achieved:

- Training accuracy: `0.979856`
- Validation accuracy: `0.978968`
- Validation F1: `0.981395`
- Validation ROC-AUC: `0.979096`
- Test accuracy: `0.979766`

### Unscaled SVM

The unscaled SVM achieved:

- Training accuracy: `0.942525`
- Validation accuracy: `0.942709`
- Validation F1: `0.949533`
- Validation ROC-AUC: `0.941520`
- Test accuracy: `0.942548`

The difference is clear. Scaling improved validation accuracy by roughly 3.6 percentage points and improved ROC-AUC substantially.

This experiment supports the decision to use `MinMaxScaler` for the normal preprocessing pipeline. Since SVM depends on distances and margins, features with different numerical scales can disproportionately influence the model.

The inference times were both very low, around 0.004 seconds, so the main difference between the two configurations was predictive performance rather than speed.

---

## 11. Training-time comparison

Training time varied considerably between models.

The fastest training model in the supplied results was **KNN K=3**, at approximately `0.019` seconds, followed closely by KNN K=15 at `0.023` seconds. Logistic Regression and the Decision Trees also trained in approximately one second.

Random Forest took approximately `27.36` seconds, while Gradient Boosting took approximately `105.90` seconds.

This shows an important trade-off: the model with the strongest predictive performance is not necessarily the cheapest to train. Gradient Boosting produced excellent results, but its training cost was much higher than the simpler models.

---

## 12. Inference-time comparison

Inference time showed an even larger difference between some models.

The supplied results show very low inference times for Logistic Regression, both SVM versions, and the Decision Trees. The deep Decision Tree required only about `0.0067` seconds, while the shallow tree required about `0.0088` seconds.

In contrast, KNN was much slower because prediction requires comparing new observations against stored training examples. KNN K=3 took approximately `33.67` seconds and KNN K=15 took approximately `41.18` seconds.

Therefore, if inference speed were an important requirement, I would not select KNN despite its strong predictive scores.

---

## 13. Bias and variance observations

The Decision Tree experiments provided the clearest view of the bias-variance trade-off.

The shallow tree was deliberately constrained to a depth of 3. Its training accuracy was `0.991512` and validation accuracy was `0.992183`, so it did not show a major train-validation gap. Its limitation is mainly its restricted complexity.

The deep tree reached `1.000000` training accuracy. This indicates that the model had enough capacity to fit the training data almost perfectly. However, unlike a typical severe overfitting case, its validation and test scores also remained extremely high.

The Random Forest and Gradient Boosting models also performed strongly on both training and validation data. Their small train-validation differences suggest that the high performance was not simply caused by a large loss of generalization.

The SVM scaling experiment demonstrates another type of limitation: the unscaled SVM had substantially weaker performance, showing that an unsuitable feature representation can introduce a form of model limitation even when the algorithm itself is capable of performing well.

---

## 14. Which model performed best?

Based on the supplied validation and test results, **Gradient Boosting** is the strongest overall model when considering balanced predictive metrics.

It achieved the highest validation ROC-AUC in the CSV at `0.999988`, validation accuracy of `0.998548`, F1 of `0.998720`, and test accuracy of `0.998488`.

However, **Deep Decision Tree** achieved the highest test accuracy at `0.999773` and had an extremely low inference time of approximately `0.0067` seconds.

Therefore, the choice depends on the objective:

- If the main objective is strong overall predictive performance across validation metrics, I would choose **Gradient Boosting**.
- If the main objective is maximum test accuracy together with very fast inference, the **Deep Decision Tree** is attractive.
- If simplicity and a strong baseline are more important, **Logistic Regression** is a reasonable choice.
- If explainability is the priority, the **Shallow Decision Tree** is easier to understand than the deeper models.

---

## 15. Final model choice

For this benchmark, I would select **Gradient Boosting as the final model** if predictive performance is the primary objective.

The reason is that it provides an excellent balance across validation accuracy, precision, recall, F1, and ROC-AUC, and it also maintains very high test accuracy. The validation metrics are not based on one score only; the model performs strongly across several evaluation measures.

I would not select it solely because it has the highest ROC-AUC. The decision is supported by its high F1-score, high precision and recall, and strong test accuracy.

The main drawback is training time. At approximately `105.90` seconds, it is much more computationally expensive than Logistic Regression or a Decision Tree. If training cost were a major constraint, I would consider the Decision Tree or Logistic Regression instead.

---

## 16. Overall conclusion

The experiments show that model selection should consider more than one performance metric.

The most important observations are:

1. **Gradient Boosting provided the strongest overall validation performance** in the supplied results and also achieved very high test accuracy.
2. **The deep Decision Tree achieved the highest test accuracy**, but its perfect training accuracy shows that its complexity should be monitored.
3. **Random Forest was also highly effective**, with strong validation accuracy, recall, F1, and ROC-AUC.
4. **Logistic Regression was a strong baseline** and showed consistent training, validation, and test performance.
5. **Increasing K in KNN produced a smoother model**, and K=15 achieved a higher validation ROC-AUC than K=3, although KNN inference was very slow.
6. **Feature scaling was important for SVM**. The scaled SVM clearly outperformed the unscaled version.
7. **Training and inference cost matter**. Gradient Boosting was the most expensive to train, while KNN was by far the most expensive at inference time.
8. **The Decision Tree experiment demonstrated the effect of model complexity**, with the shallow and deep configurations behaving differently.

Overall, the experiments were useful not just for identifying a high-performing model, but for understanding why different algorithms behaved differently. The results support Gradient Boosting as the best overall predictive choice for this benchmark, while also showing that the final choice could change if interpretability, training time, or inference speed became the main requirement.



## 17. Required Observations — Final Answers

### 1. Which model performed best on training data?

The **Deep Decision Tree (Depth=20)** performed best on the training data. It achieved a training accuracy of **1.000000**, which was the highest training accuracy among all tested models. This means the model was able to fit the training data almost perfectly. However, a perfect training score also means that model complexity needs to be considered carefully because it can be a sign of high variance or overfitting.

### 2. Which model performed best on validation data?

For the validation set, **Gradient Boosting** performed best when considering the overall combination of validation metrics. It achieved **0.998548 validation accuracy**, **0.998667 precision**, **0.998774 recall**, **0.998720 F1-score**, and **0.999988 ROC-AUC**. These results show that Gradient Boosting was consistently strong rather than relying on only one metric.

The Deep Decision Tree also performed extremely well, with **0.999803 validation accuracy**, but Gradient Boosting had the highest validation ROC-AUC and a very strong balance between precision and recall. Therefore, I consider Gradient Boosting the strongest overall validation model.

### 3. Which model generalized best to the final test set?

Based strictly on **test accuracy**, the **Deep Decision Tree** generalized best to the final test set. It achieved a test accuracy of **0.999773**, which was the highest test accuracy in the results CSV. Gradient Boosting followed with **0.998488**, while Random Forest achieved **0.995130**.

Therefore, if the question is specifically asking which model had the best final test accuracy, the answer is **Deep Decision Tree**. Its result is especially strong because its validation accuracy was also **0.999803**, showing that the model remained highly accurate on unseen data.

### 4. Which algorithm was most interpretable?

The **Decision Tree** was the most interpretable algorithm. A Decision Tree makes predictions using a sequence of decision rules, so the reasoning behind a prediction can be followed from the root of the tree to a final leaf.

Among the two Decision Tree versions, the **Shallow Decision Tree (Depth=3)** is more interpretable because it contains fewer levels and simpler rules. The Deep Decision Tree achieved better predictive results, but its larger number of decisions makes it harder for a person to understand and explain.

### 5. Which algorithm was fastest at inference time?

According to the results CSV, the **unscaled SVM** was the fastest at inference time, with an inference time of approximately **0.004088 seconds**. The scaled SVM was very close at **0.004361 seconds**, followed by Logistic Regression at **0.004815 seconds**.

Therefore, the answer based on the measured inference time is **SVM (Unscaled)**. However, the unscaled SVM had considerably weaker predictive performance than the scaled SVM, so being the fastest does not automatically make it the best model.

### 6. Which model would you choose if explainability were a requirement?

If explainability were the main requirement, I would choose the **Shallow Decision Tree (Depth=3)**. It is much easier to explain than an ensemble model such as Random Forest or Gradient Boosting because its decisions can be represented as a small number of if-then rules.

Although the Deep Decision Tree performed better, the shallow tree provides a better balance between interpretability and predictive performance. Its validation accuracy was **0.992183**, and its validation recall was **1.000000**, so it still performed very well while remaining easier to understand.

### 7. Which model would you choose if predictive performance were the primary objective?

If predictive performance were the primary objective, I would choose **Gradient Boosting** as the overall model. It produced the strongest combination of validation metrics: **0.998548 accuracy**, **0.998667 precision**, **0.998774 recall**, **0.998720 F1-score**, and **0.999988 ROC-AUC**. It also achieved a very high test accuracy of **0.998488**.

The Deep Decision Tree had the highest test accuracy at **0.999773**, so it is also a very strong candidate. However, I would choose Gradient Boosting when considering the overall validation performance across multiple metrics rather than selecting a model based on test accuracy alone.

The main disadvantage of Gradient Boosting is its training cost. It required approximately **105.90 seconds** to train, which was much longer than the simpler models. Therefore, this choice assumes that predictive performance is more important than training time.

### 8. Did any model show signs of high bias or high variance?

Yes. The experiments show differences in model complexity and provide evidence of both lower-complexity behavior and possible high variance.

The **Shallow Decision Tree** was intentionally configured with a maximum depth of 3, making it the restricted/low-complexity model. Its training accuracy was **0.991512** and validation accuracy was **0.992183**. Because its training and validation scores are very close, there is no strong train-validation gap, but the restricted depth limits how complex a decision boundary it can learn.

The clearest high-variance warning comes from the **Deep Decision Tree**. It achieved **1.000000 training accuracy**, meaning it fitted the training data perfectly. Its validation and test results were also extremely high, so the results do **not** show severe generalization failure. Nevertheless, the perfect training score indicates that the model has enough capacity to fit the training data extremely closely, so it should still be monitored for overfitting.

The **KNN K=3** model also had a higher training accuracy (**0.993771**) than validation accuracy (**0.987858**), while increasing K to 15 reduced the training accuracy to **0.988045**. This is consistent with the idea that a smaller K produces a more locally sensitive model and a larger K produces a smoother decision boundary.

Overall, the experiments demonstrate the bias-variance trade-off through the controlled shallow/deep Decision Tree and KNN experiments. However, because the validation and test performance of the deep tree remained extremely high, I would describe it as a **potential high-variance model rather than claiming that it definitely overfit the final test set**.

---

## 18. Short Summary of the Required Answers

| Question | Answer |
|---|---|
| Best training model | **Deep Decision Tree (Depth=20)** — Train Accuracy = 1.000000 |
| Best validation model | **Gradient Boosting** — strongest overall validation metrics, ROC-AUC = 0.999988 |
| Best final test model | **Deep Decision Tree (Depth=20)** — Test Accuracy = 0.999773 |
| Most interpretable algorithm | **Decision Tree**, especially the shallow tree |
| Fastest inference | **Unscaled SVM** — 0.004088 s |
| Choice for explainability | **Shallow Decision Tree (Depth=3)** |
| Choice for predictive performance | **Gradient Boosting** |
| Bias/variance | **Shallow tree = restricted complexity; Deep tree = potential high variance; KNN K=3 more locally sensitive** |

These conclusions are based on the numerical results in the supplied benchmark CSV. In particular, the distinction between “best validation model” and “best final test model” is intentional: Gradient Boosting had the strongest overall validation metrics, while the Deep Decision Tree had the highest final test accuracy.
