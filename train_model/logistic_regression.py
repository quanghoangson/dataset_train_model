from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt

# Function for Logistic Regression model
def perform_logistic_regression(feature_variables, target_variable):
    df_encoded = st.session_state.df.copy()

    for var in [target_variable] + feature_variables:
        if df_encoded[var].dtype == 'object':
            label_encoder = LabelEncoder()
            df_encoded[var] = label_encoder.fit_transform(df_encoded[var])

    X = df_encoded[feature_variables]
    y = df_encoded[target_variable]

    # Debugging: Check if the data is correctly populated
    # st.write("X (features):", X.head())
    # st.write("y (target):", y.head())

    if X.empty or y.empty:
        st.error("Lỗi: Dữ liệu trống sau khi mã hóa. Vui lòng kiểm tra lại biến mục tiêu và biến đầu vào.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Debugging: Check the split data
    # st.write("X_train:", X_train.head())
    # st.write("X_test:", X_test.head())
    # st.write("y_train:", y_train.head())
    # st.write("y_test:", y_test.head())

    scaler = StandardScaler()

    # Check if X_train is not empty
    if X_train.empty:
        st.error("Lỗi: X_train trống sau khi chia dữ liệu. Vui lòng kiểm tra lại quá trình chuẩn bị dữ liệu.")
        return

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    display_logistic_regression_results(model, X_test_scaled, y_test)

def display_logistic_regression_results(model, X_test_scaled, y_test):
    # Calculate accuracy score
    accuracy = model.score(X_test_scaled, y_test)
    st.write("Accuracy:", accuracy)
    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    auc = roc_auc_score(y_test, model.predict(X_test_scaled))

    # Display confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test_scaled))
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, cmap="Blues")
    plt.title('Logistic Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    # Display ROC Curve
    fig_roc = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Display plots in Streamlit
    st.pyplot(fig_cm)
    st.pyplot(fig_roc)

