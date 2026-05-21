import numpy as np
import pandas as pd
import pickle, os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import load_and_merge, engineer_features, get_feature_matrix

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

def evaluate_models(X_train, X_test, y_train, y_test):
    results = []
    trained = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)})
        trained[name] = model
    return pd.DataFrame(results).sort_values("R2", ascending=False), trained

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return pd.DataFrame()
    df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    return df.sort_values("Importance", ascending=False).head(15)

def train_and_save(filepath="EduPro_Online_Platform.xlsx", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    df_raw, transactions = load_and_merge(filepath)
    df_raw, df_enc = engineer_features(df_raw)

    all_results = {}
    all_importances = {}
    best_models = {}

    for target in ["EnrollmentCount", "TotalRevenue"]:
        X, y = get_feature_matrix(df_enc, target)
        feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        results_df, trained_models = evaluate_models(X_train_s, X_test_s, y_train, y_test)
        all_results[target] = results_df

        best_name = results_df.iloc[0]["Model"]
        best_model = trained_models[best_name]
        best_models[target] = {"model": best_model, "scaler": scaler,
                                "features": feature_names, "best_name": best_name}

        imp = get_feature_importance(best_model, feature_names)
        all_importances[target] = imp

        with open(f"{save_dir}/{target}_model.pkl", "wb") as f:
            pickle.dump(best_models[target], f)

    # Category revenue aggregation
    df_raw["TotalRevenue"] = df_enc["TotalRevenue"] if "TotalRevenue" in df_enc.columns else df_raw["TotalRevenue"]
    cat_revenue = df_raw.groupby("CourseCategory")["TotalRevenue"].sum().reset_index()
    cat_revenue.columns = ["Category", "Revenue"]

    with open(f"{save_dir}/metadata.pkl", "wb") as f:
        pickle.dump({
            "results": all_results,
            "importances": all_importances,
            "cat_revenue": cat_revenue,
            "df_raw": df_raw,
        }, f)

    print("Training complete.")
    for t, df in all_results.items():
        print(f"\n=== {t} ===")
        print(df.to_string(index=False))

    return all_results, all_importances, best_models, cat_revenue, df_raw

if __name__ == "__main__":
    train_and_save()
