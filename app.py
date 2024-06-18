import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import xlsxwriter
import sys
sys.path.append('./train_model')
# from decision_tree import perform_decision_tree

# from knn import decision_tree,knn, linear_regression,logistic_regression,random_forest
# from linear_regression import decision_tree,knn, linear_regression,logistic_regression,random_forest
# from logistic_regression import decision_tree,knn, linear_regression,logistic_regression,random_forest
# from random_forest import decision_tree,knn, linear_regression,logistic_regression,random_forest

class StreamlitRouter:
    def __init__(self):
        self.routes = {}
        
    def add_route(self, path, func):
        self.routes[path] = func
        
    def route(self, path):
        if path in self.routes:
            self.routes[path]()
        else:
            st.write('')

app = StreamlitRouter()



uploaded_file = st.sidebar.file_uploader("Chọn tệp", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        
    if 'df' not in st.session_state:
        st.session_state.df = df
    else:
       
        st.session_state.df = st.session_state.df 
    st.write("## Dataset")
    st.write( st.session_state.df)

    
df_copy = None
if 'df' in st.session_state:
    df_copy = st.session_state.df

# THONG TIN
def info():
    if df_copy is not None:
        num_rows, num_cols = df_copy.shape
        st.write("## Thông tin dữ liệu")
        st.write(f"Số dòng: {num_rows}, Số cột: {num_cols}")
        st.write("Thông tin về dữ liệu:")
        st.write(df_copy.describe())
        st.write("Các cột có giá trị thiếu:")
        st.write(df_copy.isna().sum())
        st.write("Các cột kiểu số:")
        st.write(df_copy.select_dtypes(include=[np.number]).columns.tolist())

        non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns.tolist()
        st.write("Các cột không phải kiểu số:")
        st.write(non_numeric_cols)
    else:
        st.warning("Không có dữ liệu để hiển thị")

def view_data():
    st.subheader("Xem dữ liệu")
    if df_copy is not None:
        st.write("10 dòng ngẫu nhiên:")
        st.write(df_copy.sample(10))
        st.write("10 dòng đầu tiên:")
        st.write(df_copy.head(10))
        st.write("10 dòng cuối cùng:")
        st.write(df_copy.tail(10))
    else:
        st.warning("Không có dữ liệu để hiển thị")

# XÓA NULL
def handle_null_values():
    st.subheader("Xử lý Null")
    
    if 'df' in st.session_state and df_copy is not None:

        st.write("Danh sách null:")
        st.write(df_copy.isnull().sum())
        
        
        columns = df_copy.columns
        selected_column = st.selectbox("Chọn cột để xử lí null:", columns)
        
        def replace_null_with_mean(column):
            if df_copy[column].dtype.kind in 'bifc':
                mean_value = df_copy[column].mean()
                df_copy[column].fillna(mean_value, inplace=True)
                st.success(f"Đã thay thế null bằng giá trị trung bình cho cột '{column}'")
            else:
                st.warning(f"Cột '{column}' không phải kiểu dữ liệu số, không thể thực hiện thay thế bằng giá trị trung bình.")
        
        def replace_null_with_mode(column):
            if df_copy[column].dtype.kind in 'OSU':
                mode_value = df_copy[column].mode().iloc[0]
                df_copy[column].fillna(mode_value, inplace=True)
                st.success(f"Đã thay thế null bằng giá trị xuất hiện nhiều nhất cho cột '{column}'")
            else:
                st.warning(f"Cột '{column}' không phải kiểu dữ liệu phù hợp, không thể thực hiện thay thế bằng giá trị xuất hiện nhiều nhất.")
        
        # Xử lí null
        if st.button("Xóa null"):
            df_copy.dropna(subset=[selected_column], inplace=True)
            df_copy.reset_index(drop=True, inplace=True)
            st.experimental_rerun()
            st.success(f"Đã xóa null cho cột '{selected_column}'")
        elif st.button("Thay thế bằng giá trị trung bình"):
            replace_null_with_mean(selected_column)
            df_copy.reset_index(drop=True, inplace=True)  
            st.experimental_rerun()
        elif st.button("Thay thế bằng giá trị xuất hiện nhiều nhất"):
            replace_null_with_mode(selected_column)
            df_copy.reset_index(drop=True, inplace=True)  
            st.experimental_rerun()
        
    else:
        st.warning("Không có dữ liệu để xử lý")

# CHUẨN HÓA ÉP KIỂU 
# 1.ngày giờ
def transform_date_column():
    st.subheader("Chuyển đổi cột ngày giờ")
    if df_copy is not None:
        date_column = st.selectbox("Chọn cột ngày giờ", df_copy.columns)
        if st.button("Chuyển đổi"):
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            st.write("Dữ liệu sau khi chuyển đổi cột ngày giờ:")
            st.write(df_copy)
    else:
        st.warning("Không có dữ liệu để chuyển đổi")

# 2.Hàm chuẩn hóa dữ liệu
def normalize_data():
    st.subheader("Chuẩn hóa dữ liệu")
    if df_copy is not None:
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        norm_option = st.selectbox("Chọn phương pháp chuẩn hóa", ["StandardScaler", "MinMaxScaler"])
        columns_to_normalize = st.multiselect("Chọn các cột để chuẩn hóa", numeric_columns)

        if st.button("Chuẩn hóa"):
            if norm_option == "StandardScaler":
                scaler = StandardScaler()
            elif norm_option == "MinMaxScaler":
                scaler = MinMaxScaler()

            df_copy[columns_to_normalize] = scaler.fit_transform(df_copy[columns_to_normalize])
            st.write("Dữ liệu sau khi chuẩn hóa:")
            st.write(df_copy)
    else:
        st.warning("Không có dữ liệu để chuẩn hóa")

# 3.Hàm đổi kiểu dữ liệu
def change_data_type():
    st.subheader("Đổi kiểu dữ liệu")
    if df_copy is not None:
        text_columns = df_copy.columns.tolist()
        column_to_encode = st.selectbox("Chọn cột để đổi kiểu dữ liệu", text_columns)
        st.write(df_copy[column_to_encode].dtype)
        data_types = ['int64', 'float64', 'object', 'bool', 'datetime64[ns]', 'timedelta64[ns]', 'category']
        column_to_encode_new = st.selectbox("Chọn kiểu dữ liệu mới", data_types)

        if st.button("Thay đổi kiểu dữ liệu"):
            try:
                if column_to_encode_new == 'datetime64[ns]':
                    df_copy[column_to_encode] = pd.to_datetime(df_copy[column_to_encode],inplace=True)
                   
                elif column_to_encode_new == 'timedelta64[ns]':
                    df_copy[column_to_encode] = pd.to_timedelta(df_copy[column_to_encode] ,inplace=True)
                    
                elif column_to_encode_new == 'category':
                    df_copy[column_to_encode] = df_copy[column_to_encode].astype('category')
                    df_copy[column_to_encode] = df_copy[column_to_encode].cat.codes
                    
                else:
                    df_copy[column_to_encode] = df_copy[column_to_encode].astype(column_to_encode_new)
                    
                st.experimental_rerun()
                st.write("Dữ liệu sau khi đổi kiểu dữ liệu:")
                st.write(df_copy)
            except Exception as e:
                st.error(f"Lỗi: {e}")
            st.session_state.df_copy 
            
    else:
        st.warning("Không có dữ liệu để đổi kiểu dữ liệu")

## XÓA CỘT
def handle_column_deletion():
    if df_copy is not None and df_copy is not None:  
            column_to_delete = st.selectbox("Chọn cột để xóa:", df_copy.columns.tolist(), key="delete_column_dropdown")
            if st.button("Xác nhận xóa cột"):
                df_copy.drop(columns=[column_to_delete], inplace=True)
                st.write(f"Đã xóa cột {column_to_delete}. Dữ liệu sau khi xóa cột:")
                st.write(df_copy.head())
                st.experimental_rerun()
    else:
        st.warning("Không có dữ liệu để xóa cột hoặc dữ liệu không tồn tại")

# KIỂM TRA TRÙNG LẶP
def handle_duplicate_data():
    if df_copy is not None and not df_copy.empty: 
        duplicate_rows = df_copy[df_copy.duplicated()]
        if duplicate_rows.empty:
            st.write("Không có hàng nào bị trùng lặp.")
        else:
            st.write("Các hàng bị trùng lặp:")
            st.write(duplicate_rows)
            if st.button("Xóa hàng trùng lặp"):
                df_copy.drop_duplicates(inplace=True)
                st.session_state.columns = df_copy.columns.tolist()  # Cập nhật danh sách cột
                st.session_state.file = None  # Xóa tệp đã tải lên để tránh lỗi về phiên bản cũ của DataFrame
                st.write("Đã xóa các hàng trùng lặp.")
                st.write(duplicate_rows)
                st.experimental_rerun()
    else:
        st.warning("Không có dữ liệu để xử lí trùng lặp hoặc dữ liệu không tồn tại")

# xử lí ngoại lệ
def exception_handling():
    if df_copy is not None:
            st.dataframe(df_copy.describe())
            st.write("Các hàng chứa giá trị ngoại lệ:")
            outlier_columns = st.selectbox("Chọn cột", df_copy.columns)
            outlier_values = st.text_input("Giá trị ngoại lệ")
            if st.button("Xóa các hàng bị giá trị ngoại lệ"):
                outlier_value = float(outlier_values)
                df_copy.drop(index=df_copy[df_copy[outlier_columns] > outlier_value].index, inplace=True)
                st.experimental_rerun()
                st.session_state.df_copy = df_copy
           

#LƯU

def save_file():
    if df_copy is not None and df_copy is not None:  
        file_format = st.selectbox("Chọn định dạng file để lưu:", ["CSV", "Excel"], key="file_format_dropdown")
        buffer = BytesIO()
        if file_format == "CSV":
            df_copy.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Tải xuống file CSV",
                data=buffer,
                file_name="data.csv",
                mime="text/csv"
            )
        elif file_format == "Excel":
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_copy.to_excel(writer, index=False, sheet_name='Sheet1')
                
            buffer.seek(0)
            st.download_button(
                label="Tải xuống file Excel",
                data=buffer,
                file_name="data.xlsx",
                mime="application/vnd.ms-excel"
            )
    else:
        st.warning("Không có dữ liệu để lưu hoặc dữ liệu không tồn tại")



def diagram():
    if df_copy is not None and df_copy is not None:  
        optison_diagram = st.selectbox("Chọn biểu đồ", ["Biểu đồ cột", "biểu đồ tròn"], key="diagram_plot")
        
        if optison_diagram == "Biểu đồ cột":
            st.subheader("Biểu đồ cột")
            if df is not None:
                target_variable = st.selectbox("Chọn biến để vẽ biểu đồ cột:", df.columns.tolist())
                data = df_copy[target_variable].value_counts().head(10)  # Take top 10 most frequent values
                fig, ax = plt.subplots(figsize=(10, 6))
                data.plot(kind='bar', ax=ax)
                plt.title('Biểu đồ cột')
                plt.xlabel(target_variable)
                plt.ylabel('Số lượng')
                st.pyplot(fig)
            else:
                st.warning("Không có dữ liệu để vẽ biểu đồ cột")

        elif optison_diagram == "biểu đồ tròn":
            st.subheader("Biểu đồ tròn")
            if df is not None:
                target_variable = st.selectbox("Chọn biến để vẽ biểu đồ tròn:", df.columns.tolist())
                data = df[target_variable].value_counts().head(10)  # Take top 10 most frequent values
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(data, labels=data.index, autopct='%1.1f%%')
                plt.title('Biểu đồ tròn')
                st.pyplot(fig)
            else:
                st.warning("Không có dữ liệu để vẽ biểu đồ tròn")
                
    else:
        st.warning("Không có dữ liệu để lưu hoặc dữ liệu không tồn tại")
    


def matrix():
    st.write("Ma trận tương quan:")
    if st.session_state['df'] is not None:
        numeric_columns = st.session_state['df'].select_dtypes(include=['float', 'int']).columns
        corr_matrix = st.session_state['df'][numeric_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()               


def train_model():
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Vui lòng tải dữ liệu trước khi huấn luyện mô hình.")
        return

    st.subheader("Huấn luyện mô hình")

    # Select model type
    model_type = st.selectbox("Chọn loại mô hình:", ["Hồi quy tuyến tính", "Hồi quy logistic", "KNN", "Decision Tree", "Random Forest"], key="model_type")

    # Select target variable and feature variables
    target_variable = st.selectbox("Chọn biến mục tiêu:", st.session_state.df.columns.tolist(), key="target_variable")
    feature_variables = st.multiselect("Chọn biến đầu vào:", st.session_state.df.columns.tolist(), key="feature_variables")
    print('feature_variables')

    if not feature_variables or not target_variable:
        st.warning("Vui lòng chọn biến đầu vào và biến đầu ra.")
        return

    # Continue with model training logic based on selected model type
    if model_type == "Hồi quy tuyến tính":
        perform_linear_regression(feature_variables, target_variable)
    elif model_type == "Hồi quy logistic":
        perform_logistic_regression(feature_variables, target_variable)
    elif model_type == "KNN":
        perform_knn_classification(feature_variables, target_variable)
    elif model_type == "Decision Tree":
        # perform_decision_tree(feature_variables, target_variable)
        perform_decision_tree(feature_variables, target_variable)
    elif model_type == "Random Forest":
        perform_random_forest(feature_variables, target_variable)

# Function for Linear Regression model
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
        st.write(f"R-squared: {r_squared:.2f}")
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

   

    # st.subheader("Dự đoán giá trị")

    # Create input fields for user to enter feature values
    # input_values = []
    # for i, feature in enumerate(feature_variables):
    #     input_value = st.text_input(f"Nhập giá trị cho {feature}", key=f"input_{i}")
    #     input_values.append(float(input_value) if input_value else 0.0)

    # # Predict button
    # predict_button = st.button("Dự đoán giá trị")
    # if predict_button:
    #     input_array = np.array(input_values).reshape(1, -1)
    #     prediction = st.session_state.model.predict(input_array)
    #     st.write(f"Giá trị dự đoán cho {target_variable}: {prediction[0]:.2f}")

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

    # # Display ROC Curve
    # fig_roc = plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.grid(True)

    # Display plots in Streamlit
    st.pyplot(fig_cm)
    # st.pyplot(fig_roc)


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


options = st.sidebar.selectbox("Chọn chức năng", [' ',"Thông tin dữ liệu", "Xem dữ liệu","Xử lí dữ liệu" ])

if options == " ":
    st.write("")
elif options == "Thông tin dữ liệu":
    info()
    st.empty()
elif options == "Xem dữ liệu":

    view_data()
    st.empty()
elif options == "Xử lí dữ liệu":
    print('ABC1')
    function_option = st.sidebar.selectbox("Xử lí dữ liệu", [' ',"Xử lý null", "Chuẩn hóa và ép kiểu dữ liệu", "Xóa cột", "Kiểm tra trùng lặp", 'Xử lí ngoại lệ'])
    if function_option == "Xử lý null":
        handle_null_values()
       
    elif function_option == "Chuẩn hóa và ép kiểu dữ liệu":
        print('ABC')
        transform_option = st.selectbox("Chọn phép biến đổi", ["Chuyển đổi cột ngày giờ","Chuẩn hóa dữ liệu","Đổi kiểu dữ liệu"])

        if transform_option == "Chuyển đổi cột ngày giờ":
            transform_date_column()
        elif transform_option == "Chuẩn hóa dữ liệu":
            normalize_data()
        elif transform_option == ("Đổi kiểu dữ liệu"):
            change_data_type()

    elif function_option == "Xóa cột":
        handle_column_deletion()
       
    elif function_option == "Kiểm tra trùng lặp":
        handle_duplicate_data()
    elif function_option == "Xử lí ngoại lệ":
        exception_handling()


    




app.add_route('/matrix', matrix)
app.add_route('/diagram', diagram)
app.add_route('/train_model', train_model)
app.add_route('/save', save_file)

st.sidebar.title("Biểu đồ - Train Model")
if st.sidebar.button('Ma trận tương quan'):
    st.session_state.page = 'matrix'

if st.sidebar.button('Biểu đồ'):
    st.session_state.page = 'diagram'
            
if st.sidebar.button('Huấn luyện mô hình'):
    st.session_state.page = 'train_model'
if st.sidebar.button('Lưu dataset'):
    st.session_state.page = 'save'
    
        
    
  
        



st.markdown(
    """
    <style>
    div.stButton > button {
        width: 100%;
        margin: 5px 0;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
app.route('/' if 'page' not in st.session_state else f'/{st.session_state.page}')