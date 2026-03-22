from src.preprocessing.preprocess import preprocess_pipeline
from src.models.remediation_multilabel_model import RemediationModel

DATA_PATH = "data/raw/GUIDE_Train.csv"

TARGET_COLUMNS = ["ActionGrouped", "ActionGranular"]

X, y, _, _ = preprocess_pipeline(DATA_PATH, TARGET_COLUMNS)

model = RemediationModel()
model.train(X, y)

y_pred = model.predict(X)

model.evaluate(y, y_pred)