# main.py
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import streamlit as st
import joblib

# ------------------- Streamlit App -------------------
st.title("Indian Currency Recognition")

# Upload dataset
uploaded_zip = st.file_uploader("Upload demo_currency_dataset.zip", type="zip")
if uploaded_zip:
    import zipfile
    import tempfile

    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(temp_dir.name)

    DATA_DIR = os.path.join(temp_dir.name, "demo_currency_dataset")
    st.success(f"Dataset extracted to {DATA_DIR}")

    # ------------------- Load & preprocess images -------------------
    X, y = [], []
    target_size = (64, 64)

    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            if not os.path.isfile(img_path):
                continue
            try:
                img = imread(img_path, as_gray=True)
                img_resized = resize(img, target_size, anti_aliasing=True)
                features = hog(
                    img_resized,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys"
                )
                X.append(features)
                y.append(label)
            except Exception as e:
                st.warning(f"Skipping {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        st.error("No images loaded! Check the dataset structure.")
    else:
        st.success(f"Loaded {len(X)} images.")

        # ------------------- Train/Test Split -------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ------------------- Balance classes -------------------
        classes = np.unique(y_train)
        max_count = max([np.sum(y_train == cls) for cls in classes])
        X_train_balanced, y_train_balanced = [], []
        for cls in classes:
            X_cls = X_train[y_train == cls]
            y_cls = y_train[y_train == cls]
            X_resampled, y_resampled = resample(X_cls, y_cls,
                                                replace=True,
                                                n_samples=max_count,
                                                random_state=42)
            X_train_balanced.append(X_resampled)
            y_train_balanced.append(y_resampled)

        X_train = np.vstack(X_train_balanced)
        y_train = np.hstack(y_train_balanced)

        # ------------------- Class weights -------------------
        class_weights_dict = dict(zip(
            classes,
            compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        ))

        # ------------------- Models -------------------
        models = {
            "SVC_rbf": (SVC(probability=True, class_weight=class_weights_dict), {
                "clf__C": [1, 10],
                "clf__gamma": ["scale", 0.01]
            }),
            "RandomForest": (RandomForestClassifier(random_state=42, class_weight=class_weights_dict), {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 20]
            }),
            "KNN": (KNeighborsClassifier(), {
                "clf__n_neighbors": [3, 5]
            })
        }

        results = []
        best_model = None
        best_acc = 0

        st.info("Training models... this may take a few minutes.")

        for name, (clf, grid) in models.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=42)),
                ("clf", clf)
            ])
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            cv_mean = gs.best_score_
            test_acc = accuracy_score(y_test, best_pipe.predict(X_test))
            results.append({"Model": name, "CV Mean": cv_mean, "Test Acc": test_acc})

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = best_pipe

        # ------------------- Show Results -------------------
        st.subheader("Model Performance")
        df_results = pd.DataFrame(results)
        df_results["CV Mean"] = df_results["CV Mean"].apply(lambda x: f"{x*100:.2f}%")
        df_results["Test Acc"] = df_results["Test Acc"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(df_results)

        # ------------------- Classification Report -------------------
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.applymap(lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x)
        st.dataframe(report_df)

        # ------------------- Confusion Matrix -------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        st.dataframe(cm_df)

        # ------------------- Save Model -------------------
        save_path = st.text_input("Enter filename to save best model (e.g., currency_model.pkl)", "currency_model.pkl")
        if st.button("Save Model"):
            joblib.dump(best_model, save_path)
            st.success(f"Model saved as {save_path}")
