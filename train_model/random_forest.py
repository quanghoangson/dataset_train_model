from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt

# Function for Random Forest model
def perform_random_forest(feature_variables, target_variable):
    df_encoded = st.session_state.df.copy()

    for var in [target_variable] + feature_variables:
        if df_encoded[var].dtype == 'object':
            label_encoder = LabelEncoder()
            df_encoded[var] = label_encoder.fit_transform(df_encoded[var])

    X = df_encoded[feature_variables]
    y = df_encoded[target_variable]

    if X.empty or y.empty:
        st.error("Lỗi: Dữ liệu trống sau khi mã hóa. Vui lòng kiểm tra lại biến mục tiêu và biến đầu vào.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Display model evaluation
    display_random_forest_results(model, X_test, y_test, feature_variables)
# Function to display results for Random Forest model
def display_random_forest_results(model, X_test, y_test, feature_variables):
    # Calculate accuracy of Random Forest
    accuracy = model.score(X_test, y_test)

    # Display Random Forest Tree
    plt.figure(figsize=(12, 8))
    plot_tree(model.estimators_[0], feature_names=feature_variables, class_names=[str(x) for x in model.classes_], filled=True)
    plt.title("Random Forest")
    st.pyplot(plt.gcf())

    # Display accuracy
    st.write(f'Accuracy of Random Forest: {accuracy:.2f}')

