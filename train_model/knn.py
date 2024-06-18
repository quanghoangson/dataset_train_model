from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt

# Function for KNN Classification
def perform_knn_classification(feature_variables, target_variable):
    df_encoded = st.session_state.df.copy()

    # Encode categorical variables if necessary
    for var in [target_variable] + feature_variables:
        if df_encoded[var].dtype == 'object':
            label_encoder = LabelEncoder()
            df_encoded[var] = label_encoder.fit_transform(df_encoded[var])

    X = df_encoded[feature_variables]
    y = df_encoded[target_variable]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features (example using StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the KNN model
    model = KNeighborsClassifier()

    # Fit the model on the scaled training data
    model.fit(X_train_scaled, y_train)

    # Display results
    display_knn_classification_results(model, X_train_scaled, X_test_scaled, y_train, y_test)

def display_knn_classification_results(model, X_train_scaled, X_test_scaled, y_train, y_test):
    # Evaluate on training data
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_conf_matrix = confusion_matrix(y_train, train_predictions)
    train_classification_report = classification_report(y_train, train_predictions)

    st.subheader("Đánh giá mô hình")
    st.write(f"Train Accuracy: {train_accuracy:.2f}")
    # st.write("Confusion Matrix (Training):")
    # st.write(train_conf_matrix)
    # st.write("Classification Report (Training):")
    # st.write(train_predictions)

    # Calculate accuracies for different k values
    k_values = range(1, 11)  # Example: Consider k from 1 to 10
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        accuracy = knn.score(X_test_scaled, y_test)
        accuracies.append(accuracy)

    # Plotting the Accuracy vs. Number of Neighbors (k)
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Elbow Method for Optimal k in KNN')
    plt.grid(True)

    # Display the plot using Streamlit
    st.pyplot(plt.gcf())

    # Clear the current figure to avoid overlapping plots
    plt.clf()
