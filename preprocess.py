from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess(df):
    le = LabelEncoder()
    df['Location'] = le.fit_transform(df['Location'])
    df['Character'] = le.fit_transform(df['Character'])

    scaler = StandardScaler()
    df[['Age', 'Duration', 'Frequency']] = scaler.fit_transform(df[['Age', 'Duration', 'Frequency']])

    X = df.drop('Type', axis=1)
    y = df['Type']

    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)

    return X_res, y_res
