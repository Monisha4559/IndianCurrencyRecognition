
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
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
import joblib

# ------------------- Paths -------------------
DATA_ZIP = "demo_currency_dataset.zip"  # Path to your zip
WORK_DIR = "demo_currency_dataset"
MODEL_PATH = "currency_pipeline.pkl"
CONF_MATRIX_PATH = "confusion_matrix.png"

# ------------------- Unzip dataset -------------------
if not os.path.exists(WORK_DIR):
    with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")
print(f"✅ Dataset unzipped to {WORK_DIR}")

# ------------------- Find the correct dataset folder -------------------
dataset_dir = WORK_DIR
# If zip extracted an extra folder level, handle it
subfolders = [f for f in os.listdir(WORK_DIR) if os.path.isdir(os.path.join(WORK_DIR, f))]
if len(subfolders) == 1 and subfolders[0].startswith("demo_currency_dataset"):
    dataset_dir = os.path.join(WORK_DIR, subfolders[0])

print(f"Using dataset folder: {dataset_dir}")

# ------------------- Load & preprocess images -------------------
X, y = [], []
target_size = (64, 64)

for label in os.listdir(dataset_dir):
    folder = os.path.join(dataset_dir, label)
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
            print(f"⚠️ Skipping {img_path}, not a valid image")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("No images loaded! Check your zip file and folder structure.")
print(f"✅ Feature matrix: {X.shape}, Labels: {len(y)}")

# ------------------- Train/Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ------------------- Balance classes -------------------
classes = np.unique(y_train)
max_count = max([np.sum(y_train == cls) for cls in classes])
X_train_balanced, y_train_balanced = [], []
for cls in classes:
    X_cls = X_train[y_train == cls]
    y_cls = y_train[y_train == cls]
    X_resampled, y_resampled = resample(X_cls, y_cls, replace=True, n_samples=max_count, random_state=42)
    X_train_balanced.append(X_resampled)
    y_train_balanced.append(y_resampled)

X_train = np.vstack(X_train_balanced)
y_train = np.hstack(y_train_balanced)
print("Balanced Train:", X_train.shape)

# ------------------- Class weights -------------------
class_weights_dict = dict(zip(
    classes,
    compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
))
print("Class weights:", class_weights_dict)

# ------------------- Define models & grids -------------------
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

# ------------------- Train & Evaluate -------------------
results = []
best_model = None
best_acc = 0

for name, (clf, grid) in models.items():
    print(f"\n🔍 Training {name} ...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("clf", clf)
    ])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)

    best_pipe = gs.best_estimator_
    cv_mean = gs.best_score_
    test_acc = accuracy_score(y_test, best_pipe.predict(X_test))

    results.append({"name": name, "cv_mean": cv_mean, "test_acc": test_acc})

    if test_acc > best_acc:
        best_acc = test_acc
        best_model = best_pipe

# ------------------- Report Results -------------------
print("\n📊 Model Performance:")
print(f"{'Model':<15} {'CV Mean':<10} {'Test Acc':<10}")
for res in results:
    print(f"{res['name']:<15} {res['cv_mean']*100:>6.2f}%   {res['test_acc']*100:>6.2f}%")

joblib.dump(best_model, MODEL_PATH)
print(f"\n🏆 Best model saved: {type(best_model.named_steps['clf']).__name__}")

# ------------------- Classification Report -------------------
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
print("\n📑 Classification Report (in %):")
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"\nClass {label}:")
        for m, v in metrics.items():
            print(f"  {m}: {v*100:.2f}%")

# ------------------- Confusion Matrix -------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(CONF_MATRIX_PATH)
plt.show()
