# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import streamlit as st
    from sklearn import datasets
    from sklearn.ensemble import *
    from sklearn.ensemble import RandomForestClassifier

    # Load the iris dataset
    iris_data = datasets.load_iris()
    # Set the title
    st.title("Iris Dataset Classifier")

    # Set the header
    st.header("Random Forest Classifier")

    # Assign the data and target variables
    X = iris_data.data
    Y = iris_data.target
    # Set up a Random Forest Classifier
    rf_classifier = RandomForestClassifier()

    # Fit the model
    rf_classifier.fit(X, Y)
    # Add input fields for sepal length, sepal width, petal length, and petal width
    sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    st.text('Selected: {}'.format(sepal_length))
    sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    st.text('Selected: {}'.format(sepal_width))
    petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    st.text('Selected: {}'.format(petal_length))
    petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
    st.text('Selected: {}'.format(petal_width))
    # Define the prediction button
    if st.button("Predict"):
        # Perform prediction
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = rf_classifier.predict(input_data)

        # Display the prediction
        st.subheader("Prediction")
        st.write("Predicted Iris Flower Type:",iris_data.target_names[prediction[0]])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
