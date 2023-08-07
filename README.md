# Simplifying Model Deployment with Gradio: A Step-by-Step Guide

## Introduction:

To deploy machine learning models is a pivotal step of bringing a models into real-world applications. Tough, the process of deployment and integration of models with user interfaces can be complex and prolonged. 
Python library named Gradio, makes this process easier by providing a user-friendly interface for deploying machine learning models.   
In this article, I will help you to navigate through step-by-step process of machine learning model deployment by using Gradio.

- Step 1: Create a project folder

Firstly, create new folder for the project. let's call it gradio_app. Next step is to create a requirements.txt file which will cantain all the packages and libraries needed in virtual environment for app to be able to run, then create a readme.md file to put a few descriptions about the work then a .gitignore file with the name of your virtual environment inside for the environment folder to be ignored by git while creating a repository for the project. Also create an src folder to contain app.py file and a machine learning model component file or other files that may have been exported from notebook. Once the folder is ready,  open it in a text editor, let's use visual studio code.
Once the project folder is opened in vscode, open the requirements.txt file cantaining the required installations, then open a new terminal to install the requirements by typing pip install + the name of requirement ten enter, till all requirements are installed.


- Step 2: Prepare model

Before deploying a model, is better make sure that it is trained and ready to use. 

- Step 3: Import the necessary libraries

In app.py, import the required libraries, including Gradio and the libraries associated with machine learning framework. For example, Scikit-learn, pickle,..

```
import gradio as gr
import sklearn
import pickle
import pandas pd
import numpy as np
```

- Step 4: Define your model and input/output functions

Define your model and any necessary preprocessing functions to ensure that the input and output formats match the requirements of your model.

```
def make_prediction(gender, Partner, Dependents, tenure, MultipleLines,
       InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, PaperlessBilling, PaymentMethod,
       MonthlyCharges, TotalCharges):
   input_data = pd.DataFrame({'gender':[gender], 'Partner':[Partner], 'Dependents':[Dependents], 'tenure':[tenure], 'MultipleLines':[MultipleLines],
       'InternetService':[InternetService], 'OnlineSecurity':[OnlineSecurity], 'OnlineBackup':[OnlineBackup], 'DeviceProtection':[DeviceProtection],
       'TechSupport':[TechSupport], 'Contract':[Contract], 'PaperlessBilling':[PaperlessBilling], 'PaymentMethod':[PaymentMethod],
       'MonthlyCharges':[MonthlyCharges], 'TotalCharges':[TotalCharges]})
   
   #load already saved pipeline and make predictions
    with open("ml_model.pkl", "rb") as f:
        model = pickle.load(f)
        predt = model.predict(input_data) 
    #return prediction 
    return predt
    
```
- Step 5: Create the attribute component
```
#create the inpuut components for gradio
gender_input = gr.inputs.Dropdown(choices =['Female', 'Male']) 
Partner_input = gr.inputs.Dropdown(choices =['Yes', 'No']) 
Dependents_input = gr.inputs.Dropdown(choices =['Yes', 'No'])
tenure_input = gr.Number()
MultipleLines_input = gr.inputs.Dropdown(choices =['No phone service', 'No', 'Yes'])
InternetService_input = gr.inputs.Dropdown(choices =['DSL', 'Fiber optic', 'No']) 
OnlineSecurity_input = gr.inputs.Dropdown(choices =['No', 'Yes', 'No internet service']) 
OnlineBackup_input = gr.inputs.Dropdown(choices =['Yes', 'No', 'No internet service']) 
DeviceProtection_input = gr.inputs.Dropdown(choices =['No', 'Yes', 'No internet service'])
TechSupport_input = gr.inputs.Dropdown(choices =['No', 'Yes', 'No internet service'])
Contract_input = gr.inputs.Dropdown(choices =['Month-to-month', 'One year', 'Two year'])
PaperlessBilling_input = gr.inputs.Dropdown(choices =['Yes', 'No']) 
PaymentMethod_input = gr.inputs.Dropdown(choices =['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])    
MonthlyCharges_input = gr.Number()
TotalCharges_input = gr.Number()

output = gr.Textbox(label='Prediction') 
```

- Step 6: Create the Gradio interface

Create the Gradio interface by defining the input type, output type, and any additional parameters you want to include. Gradio supports various input types, including images, text inputs, and dropdown menus, and output types, such as text and images.

```
#create the interface component

app = gr.Interface(fn =make_prediction,inputs =[gender_input,
                                                 Partner_input,
                                                 Dependents_input,
                                                 tenure_input,
                                                 MultipleLines_input,
                                                 InternetService_input,
                                                 OnlineSecurity_input,
                                                 OnlineBackup_input,
                                                 DeviceProtection_input,
                                                 TechSupport_input,
                                                 Contract_input,
                                                 PaperlessBilling_input,
                                                 PaymentMethod_input,
                                                 MonthlyCharges_input,
                                                 TotalCharges_input],
                                                 title ="Customer Churn Predictor", 
                                                  description="Enter the feilds Below and click the submit button to Make Your Prediction",
                                                 outputs = output)

```

- Step 7: Launch the interface

Launch the Gradio interface using the `launch()` method. This will start a local web server hosting your model.

```
app.launch(share = True)
```

- Step 8: Test your deployed model

After launching the interface, you can test your deployed model by interacting with the provided web interface. 

- Step 9: Share your deployed model

Gradio provides an easy way to share your deployed model with others. By default, Gradio creates a web interface accessible at `http://localhost:7860`. You can share this URL with others, allowing them to interact with your deployed model through their web browser.

## Conclusion:

Deploying machine learning models can be a challenging task, but with Gradio, the process becomes easier. By previous step-by-step guide listed in this article, enables to quickly deploy machine learning models with a user-friendly interface. Gradio's flexibility and ease of use make it a distinct for prototyping and showcasing of models to convience audience. 
