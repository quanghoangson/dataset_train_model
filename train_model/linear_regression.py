from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt

def perform_linear_regression(feature_variables, target_variable):
    X = st.session_state.df[feature_variables]
    y = st.session_state.df[target_variable]

    model = LinearRegression()
    model.fit(X, y)

    # Display model evaluation
    display_regression_results(model, X, y, feature_variables, target_variable)
def display_regression_results(model, X, y, feature_variables, target_variable):
    y_pred = model.predict(X)
    r_squared = model.score(X, y)

    if len(feature_variables) == 1:
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, 0], y, color='blue', label='Actual')
        ax.plot(X.iloc[:, 0], y_pred, color='red', label='Predicted')
        ax.set_xlabel(feature_variables[0])
        ax.set_ylabel(target_variable)
        ax.set_title(f"Linear Regression: {target_variable} vs {feature_variables[0]}")
        ax.grid(True)
        ax.legend()
        ax.text(0.05, 0.95, f'R-squared: {r_squared:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Store the trained model in session state
        st.session_state.model = model

    else:
        # Display 3D scatter plot and predicted values
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, color='blue', label='Actual')
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y_pred, color='red', label='Predicted')

        ax.set_xlabel(feature_variables[0])
        ax.set_ylabel(feature_variables[1])
        ax.set_zlabel(target_variable)
        ax.set_title(f"Linear Regression: {target_variable} vs {', '.join(feature_variables)}")
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Store the trained model in session state
        st.session_state.model = model

    st.write(f"R-squared: {r_squared:.2f}")

    st.subheader("Dự đoán giá trị")

    # Create input fields for user to enter feature values
    input_values = []
    for i, feature in enumerate(feature_variables):
        input_value = st.text_input(f"Nhập giá trị cho {feature}", key=f"input_{i}")
        input_values.append(float(input_value) if input_value else 0.0)

    # Predict button
    predict_button = st.button("Dự đoán giá trị")
    if predict_button:
        input_array = np.array(input_values).reshape(1, -1)
        prediction = st.session_state.model.predict(input_array)
        st.write(f"Giá trị dự đoán cho {target_variable}: {prediction[0]:.2f}")
