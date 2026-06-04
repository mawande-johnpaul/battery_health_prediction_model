





param_distributions = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': [None, 'sqrt', 'log2'],
}

search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_mean_absolute_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)
best_params = search.best_params_
print(f'Best parameters: {best_params}')
print(f'Best CV MAE from search: {-search.best_score_:.2f} Days')

# Cross-validate the best tuned model on the training split before final test evaluation.
best_model = search.best_estimator_
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
print(f'Cross-val MAE: {-cv_scores.mean():.2f} +/- {cv_scores.std():.2f} Days')

model = GradientBoostingRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# ==========================================
# 4b. SHAP EXPLAINABILITY ANALYSIS
# ==========================================
print("\n--- Step 2b: SHAP Feature Importance Analysis ---")
# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train.iloc[:200])  # Use subset for speed

# SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_train.iloc[:200], plot_type='bar', show=False)
plt.title('SHAP Feature Importance - Mean |SHAP value|', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
print("SHAP feature importance plot saved as 'shap_feature_importance.png'")
plt.close()

# SHAP dependence plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, feature in enumerate(features):
    shap.dependence_plot(feature, shap_values, X_train.iloc[:200], ax=axes[idx], show=False)
plt.tight_layout()
plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
print("SHAP dependence plots saved as 'shap_dependence_plots.png'")
plt.close()

# ==========================================
# 4c. LIME LOCAL INTERPRETABILITY
# ==========================================
print("\n--- Step 2c: LIME Local Model Explanations ---")
# Create LIME explainer with a wrapper for regression prediction
def predict_fn(X):
    """Wrapper to make regression predictions suitable for LIME."""
    predictions = model.predict(X)
    # LIME expects probabilities for binary classification, but for regression
    # we can normalize predictions to a 0-1 range for visualization
    min_pred = y_train.min()
    max_pred = y_train.max()
    normalized = (predictions - min_pred) / (max_pred - min_pred)
    # Create a 2-column output (LIME internal requirement)
    return np.column_stack([1 - normalized, normalized])

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=features,
    verbose=False,
    random_state=42,
    mode='regression'
)

# Generate LIME explanations for test samples
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
test_indices = [0, 10, 20, 30]

for plot_idx, test_idx in enumerate(test_indices):
    if test_idx < len(X_test):
        # Use raw model.predict for LIME instead of the wrapper
        exp = lime_explainer.explain_instance(
            X_test.iloc[test_idx].values,
            model.predict,
            num_features=len(features)
        )
        
        # Extract explanation data
        exp_list = exp.as_list()
        feature_names_exp = [item[0] for item in exp_list]
        feature_weights = [item[1] for item in exp_list]
        
        # Plot
        colors = ['green' if w > 0 else 'red' for w in feature_weights]
        axes[plot_idx].barh(feature_names_exp, feature_weights, color=colors, alpha=0.7, edgecolor='black')
        axes[plot_idx].set_xlabel('Contribution to Prediction', fontweight='bold')
        actual_days = int(y_test.iloc[test_idx])
        predicted_days = int(model.predict(X_test.iloc[test_idx:test_idx+1])[0])
        axes[plot_idx].set_title(f'Sample {test_idx}: Actual={actual_days}d, Predicted={predicted_days}d', fontweight='bold')
        axes[plot_idx].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('lime_explanations.png', dpi=300, bbox_inches='tight')
print("LIME explanation plots saved as 'lime_explanations.png'")
plt.close()

# ==========================================
# 5. COMPILING VALIDATION PERFORMANCE
# ==========================================
print("\n--- Step 3: Assessing Performance Against Withheld Test Rows ---")
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae:.2f} Days")
print(f"Model Variance Accuracy (R² Score): {r2:.4f}")

# ==========================================
# 6. LIVE SMARTPHONE DIAGNOSTIC SIMULATION
# ==========================================
print("\n--- Simulation: Connected Device Check at Shop Counter ---")
# A technician connects a customer's phone over USB. 
# The script takes a 1-second snapshot of the system fuel gauge registers:
mock_extracted_phone = pd.DataFrame([{
    'SOH_capacity_pct': 0.84,  # Fuel gauge reports the battery is at 84% maximum health
    'avg_voltage': 3.65,       # Operational voltage sag signature under baseline benchmark load
    'avg_temp': 40.2           # Active operating core temperature in degrees Celsius
}])

predicted_days = model.predict(mock_extracted_phone[features])[0]
print(f"-> Diagnostic Output: Based on hardware wear, this device has been used for roughly {int(predicted_days)} days.")