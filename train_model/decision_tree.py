from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt

# class decision_tree:
    # Function for Decision Tree model
def perform_decision_tree(feature_variables, target_variable):
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

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Display model evaluation
    display_decision_tree_results(model, X_test, y_test, feature_variables)
# Function to display results for Decision Tree model
def display_decision_tree_results(model, X_test, y_test, feature_variables):
    # Calculate accuracy of Decision Tree
    accuracy = model.score(X_test, y_test)

    # Display Decision Tree
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=feature_variables, class_names=[str(x) for x in model.classes_], filled=True)
    plt.title("Decision Tree")
    st.pyplot(plt.gcf())

    # Display accuracy
    st.write(f'Accuracy of Decision Tree: {accuracy:.2f}')

