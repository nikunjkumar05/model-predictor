import numpy as np
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder

def cor(a, b):
    corr=np.corrcoef(a, b)[0, 1]  # Pearson correlation
    if np.isnan(corr):
        return 0.0
    return corr

def identify_model(trained_models, X_user, y_true, y_pred_uploaded, task_type, temperature=1.0):
    scores={}
    if task_type=="Classification":
        #using label encoder for categorical columns
        label_encoder=LabelEncoder()
        all_labels=np.concatenate([y_true, y_pred_uploaded])
        for model in trained_models.values():
                preds=model.predict(X_user)
                all_labels=np.concatenate([all_labels, preds])
        label_encoder.fit(all_labels)
        y_true=label_encoder.transform(y_true)
        y_pred_uploaded=label_encoder.transform(y_pred_uploaded)

        for name, model in trained_models.items():
            try:
                model_preds = model.predict(X_user)
                model_preds = label_encoder.transform(model_preds)
                model_probs = model.predict_proba(X_user)
                acc =accuracy_score(y_true, model_preds)
                f1 =f1_score(y_true, model_preds, average='weighted')
                try:
                    ll =log_loss(y_true, model_probs)
                except:
                    ll =np.inf

                similaritiy =accuracy_score(y_pred_uploaded, model_preds)
                corr =cor(y_pred_uploaded, model_preds)
                # Combine multiple classification metrics into a single weighted score:
                # -acc (15%): accuracy of the model (higher is better)
                # -f1 (10%): F1-score, accounts for precision and recall (higher is better)
                # -ll (-10%): log loss, measures prediction uncertainty (lower is better, so subtracted)
                # -similarity (65%): how close uploaded predictions are to this model’s predictions (higher is better)
                # -corr (10%): correlation between uploaded predictions and model predictions (higher is better)

                scores[name] = (0.15*acc + 0.10*f1 - 0.10*ll + 0.65*similaritiy + 0.10*corr) #I manually tried on a dataset
            except:
                scores[name] = -np.inf

    else:  
        # Regression
        for name, model in trained_models.items():
            try:
                model_preds = model.predict(X_user)
                r2 = r2_score(y_true, model_preds)
                rmse = np.sqrt(mean_squared_error(y_true, model_preds))
                mae = mean_absolute_error(y_true, model_preds)

                similarity = np.exp(-mean_squared_error(y_pred_uploaded, model_preds))
                corr = cor(y_pred_uploaded, model_preds)
                # Combine different regression metrics into a single weighted score.
                # Each coefficient (0.55,0.15,etc.) represents how important that metric is:
                # -similarity (55% weight): how close uploaded predictions are to model predictions (higher is better)
                # -corr (15% weight): correlation between uploaded predictions and model predictions (higher is better)
                # -r2 (10% weight): coefficient of determination (higher is better)
                # -rmse (10% penalty): root mean squared error (lower is better, so subtracted)
                # -mae (10% penalty): mean absolute error (lower is better, so subtracted)
                scores[name] = (0.55*similarity + 0.15*corr + 0.1*r2 - 0.1*rmse - 0.1*mae) 
            except:
                scores[name] = -np.inf

    score_array = np.array(list(scores.values()))
    # Apply temperature scaling  
    exp_scores = np.exp(score_array / temperature)
    ## Normalize so that all probabilities sum to 1
    proba = {k: v / np.sum(exp_scores) for k, v in zip(scores.keys(), exp_scores)}

    predicted_model = max(proba, key=proba.get)
    return predicted_model, proba
