from flask import Flask,render_template,request,redirect, url_for, session
import pickle
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

# Path to the static folder containing images
static_folder = 'static'

# List of image file extensions to delete
image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

# Iterate over files in the static folder
for filename in os.listdir(static_folder):
    # Check if the file is an image based on its extension
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        # Construct the full path to the file
        file_path = os.path.join(static_folder, filename)
        # Delete the file
        os.remove(file_path)



app = Flask(__name__)

global_df = None

model = pickle.load(open('xgboost_model.pkl', 'rb'))
@app.route('/')
def index():
 return render_template('main.html')

@app.route('/student')
def student():
    return render_template('index.html')

# @app.route('/teacher')
# def teacher():
#     return render_template('teacher.html')

@app.route('/predict',methods = ['POST'])
def result():
 if request.method == 'POST':
    gender = request.form['gender']
    tenth_percentage = float(request.form['10th_percentage'])
    twelfth_percentage = float(request.form['12th_percentage'])
    twelfth_stream = request.form['12th_stream']
    family_count = int(request.form['family_count'])
    father_occupation = request.form['Father Occupation']
    siblings_count = int(request.form['siblings_count'])
    english_1st_sem = float(request.form['english_1st_sem'])
    math_1st_sem = float(request.form['math_1st_sem'])
    basic_science_1st_sem = float(request.form['basic_science_1st_sem'])
    ict_1st_sem = float(request.form['ict_1st_sem'])
    wpc_1st_sem = float(request.form['wpc_1st_sem'])
    eec_2st_sem = float(request.form['eec_2st_sem'])
    ami_2st_sem = float(request.form['ami_2st_sem'])
    bec_2st_sem = float(request.form['bec_2st_sem'])
    pci_2st_sem = float(request.form['pci_2st_sem'])
    bcc_2st_sem = float(request.form['bcc_2st_sem'])
    cph_2st_sem = float(request.form['cph_2st_sem'])
    wpd_2st_sem = float(request.form['wpd_2st_sem'])
    oop_3st_sem = float(request.form['oop_3st_sem'])
    dsu_3st_sem = float(request.form['dsu_3st_sem'])
    cgr_3st_sem = float(request.form['cgr_3st_sem'])
    dms_3st_sem = float(request.form['dms_3st_sem'])
    dte_3st_sem = float(request.form['dte_3st_sem'])
    jpr_4st_sem = float(request.form['jpr_4st_sem'])
    sen_4st_sem = float(request.form['sen_4st_sem'])
    dcc_4st_sem = float(request.form['dcc_4st_sem'])
    mic_4st_sem = float(request.form['mic_4st_sem'])
    gad_4st_sem = float(request.form['gad_4st_sem'])

    gender = request.form['gender']
    
    data = pd.DataFrame({
        'Gender': [gender],
        '10th Percentage': [tenth_percentage],
        '12th Percentage': [twelfth_percentage],
        '12th Stream': [twelfth_stream],
        'Family Count': [family_count],
        'Father Occupation': [father_occupation],
        'Siblings Count': [siblings_count],
        'English 1st Sem': [english_1st_sem],
        'Math 1st Sem': [math_1st_sem],
        'Basic Science 1st Sem': [basic_science_1st_sem],
        'ICT 1st Sem': [ict_1st_sem],
        'WPC 1st Sem': [wpc_1st_sem],
        'EEC 2st Sem': [eec_2st_sem],
        'AMI 2st Sem': [ami_2st_sem],
        'BEC 2st Sem': [bec_2st_sem],
        'PCI 2st Sem': [pci_2st_sem],
        'BCC 2st Sem': [bcc_2st_sem],
        'CPH 2st Sem': [cph_2st_sem],
        'WPD 2st Sem': [wpd_2st_sem],
        'OOP 3st Sem': [oop_3st_sem],
        'DSU 3st Sem': [dsu_3st_sem],
        'CGR 3st Sem': [cgr_3st_sem],
        'DMS 3st Sem': [dms_3st_sem],
        'DTE 3st Sem': [dte_3st_sem],
        'JPR 4st Sem': [jpr_4st_sem],
        'SEN 4st Sem': [sen_4st_sem],
        'DCC 4st Sem': [dcc_4st_sem],
        'MIC 4st Sem': [mic_4st_sem],
        'GAD 4st Sem': [gad_4st_sem]
    })



    data['Gender']=np.where(data['Gender']=='Male',1,0)


    stream_12_mapping = {'Science': 2, 'Commerce': 1,'Art':0}
    data['12th Stream'] = data['12th Stream'].map(stream_12_mapping)

    data = pd.get_dummies(data)
    
    # columns_to_replace = ['English 1st Sem', 'Math 1st Sem', 'Basic Science 1st Sem', 'ICT 1st Sem', 'WPC 1st Sem', 'EEC 2st Sem', 'AMI 2st Sem', 'BEC 2st Sem', 'PCI 2st Sem', 'BCC 2st Sem', 'CPH 2st Sem', 'WPD 2st Sem', 'OOP 3st Sem', 'DSU 3st Sem', 'CGR 3st Sem', 'DMS 3st Sem', 'DTE 3st Sem', 'JPR 4st Sem', 'SEN 4st Sem', 'DCC 4st Sem', 'MIC 4st Sem', 'GAD 4st Sem']
    # data[columns_to_replace] = data[columns_to_replace].apply(lambda x: str(x).replace(' ', '_'))

    train_col=['Gender', '10th Percentage', '12th Percentage', '12th Stream', 'Family Count', 'Siblings Count', 'English_1st_Sem', 'Math_1st_Sem', 'Basic Science_1st_Sem', 'ICT_1st_Sem', 'WPC_1st_Sem', 'EEC_2st_Sem', 'AMI_2st_Sem', 'BEC_2st_Sem', 'PCI_2st_Sem', 'BCC_2st_Sem', 'CPH_2st_Sem', 'WPD_2st_Sem', 'OOP_3st_Sem', 'DSU_3st_Sem', 'CGR_3st_Sem', 'DMS_3st_Sem', 'DTE_3st_Sem', 'JPR_4st_Sem', 'SEN_4st_Sem', 'DCC_4st_Sem', 'MIC_4st_Sem', 'GAD_4st_Sem', 'Father Occupation_Army', 'Father Occupation_Business', 'Father Occupation_Farmer', 'Father Occupation_Government Job', 'Father Occupation_Housewife', 'Father Occupation_No Job', 'Father Occupation_Private Job']

    input_col=['Gender', '10th Percentage', '12th Percentage', '12th Stream', 'Family Count', 'Siblings Count', 'English_1st_Sem', 'Math_1st_Sem', 'Basic Science_1st_Sem', 'ICT_1st_Sem', 'WPC_1st_Sem', 'EEC_2st_Sem', 'AMI_2st_Sem', 'BEC_2st_Sem', 'PCI_2st_Sem', 'BCC_2st_Sem', 'CPH_2st_Sem', 'WPD_2st_Sem', 'OOP_3st_Sem', 'DSU_3st_Sem', 'CGR_3st_Sem', 'DMS_3st_Sem', 'DTE_3st_Sem', 'JPR_4st_Sem', 'SEN_4st_Sem', 'DCC_4st_Sem', 'MIC_4st_Sem', 'GAD_4st_Sem', 'Father Occupation_Government Job']
    
    data.columns=input_col
    for i in list(set(train_col)-set(data.columns.tolist())):
      data[i]=0
    
    col_act = data.columns.tolist()
    prediction= model.predict(data[train_col])
    prediction=int(prediction)
    if prediction==0:
       prediction= "Less than 35%"
    elif prediction==1:
       prediction= "35% to 50%"
    elif prediction==2:
       prediction= "50% to 75%"
    else:
       prediction="75% to 100%"   

 return render_template('index.html',prediction='Your 5th Semester Percentages will be between :{}'.format(prediction))


def generate_box_plots(df, predicted_column):
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_columns=list(set(numeric_columns)-set(['Roll No']))
    plots_info = []
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=predicted_column, y=col, data=df,order=['<35%','35% to 50%', '50% to 75%','75% to 100%'])
        plt.title(f"Boxplot of {col} vs. {predicted_column}")
        plt.xlabel(predicted_column)
        plt.ylabel(col)
        plot_filename = f"static/boxplot_{col}_vs_{predicted_column}.png"
        plt.savefig(plot_filename)
        plt.close()
        plots_info.append({'filename': plot_filename, 'description': f"Boxplot of {col} vs. {predicted_column}"})
    return plots_info



@app.route('/teacher', methods=['GET', 'POST'])
def teacher():
    global global_df

    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            pred_df=df.copy()
            pred_df.drop(columns=['Roll No','Mother Occupation'],axis=1,inplace=True)
            
            pred_df['Gender']=np.where(pred_df['Gender']=='Male',1,0)
            stream_12_mapping = {'Science': 2, 'Commerce': 1,'Art':0}
            pred_df['12th Stream'] = pred_df['12th Stream'].map(stream_12_mapping)
            pred_df = pd.get_dummies(pred_df)
            train_col=['Gender', '10th Percentage', '12th Percentage', '12th Stream', 'Family Count', 'Siblings Count', 'English_1st_Sem', 'Math_1st_Sem', 'Basic Science_1st_Sem', 'ICT_1st_Sem', 'WPC_1st_Sem', 'EEC_2st_Sem', 'AMI_2st_Sem', 'BEC_2st_Sem', 'PCI_2st_Sem', 'BCC_2st_Sem', 'CPH_2st_Sem', 'WPD_2st_Sem', 'OOP_3st_Sem', 'DSU_3st_Sem', 'CGR_3st_Sem', 'DMS_3st_Sem', 'DTE_3st_Sem', 'JPR_4st_Sem', 'SEN_4st_Sem', 'DCC_4st_Sem', 'MIC_4st_Sem', 'GAD_4st_Sem', 'Father Occupation_Army', 'Father Occupation_Business', 'Father Occupation_Farmer', 'Father Occupation_Government Job', 'Father Occupation_Housewife', 'Father Occupation_No Job', 'Father Occupation_Private Job']
            # input_col=['Gender', '10th Percentage', '12th Percentage', '12th Stream', 'Family Count', 'Siblings Count', 'English_1st_Sem', 'Math_1st_Sem', 'Basic Science_1st_Sem', 'ICT_1st_Sem', 'WPC_1st_Sem', 'EEC_2st_Sem', 'AMI_2st_Sem', 'BEC_2st_Sem', 'PCI_2st_Sem', 'BCC_2st_Sem', 'CPH_2st_Sem', 'WPD_2st_Sem', 'OOP_3st_Sem', 'DSU_3st_Sem', 'CGR_3st_Sem', 'DMS_3st_Sem', 'DTE_3st_Sem', 'JPR_4st_Sem', 'SEN_4st_Sem', 'DCC_4st_Sem', 'MIC_4st_Sem', 'GAD_4st_Sem', 'Father Occupation_Government Job']
    
            pred_df.columns=train_col        
            col_act = pred_df.columns.tolist()
            df['prediction']= model.predict(pred_df[train_col])

            target_mapping = {1:'35% to 50%', 0:'<35%', 3:'75% to 100%', 2:'50% to 75%'}
            df['prediction'] = df['prediction'].map(target_mapping)

            global_df = df  # Store the DataFrame in the global variable



            return render_template('teacher.html', table=df.to_html())
        
    return render_template('teacher.html')

@app.route('/eda', methods=['POST'])
def eda():
    global global_df
        
    if global_df is not None:
        # Assuming 'predicted_column' is the column name to predict
        predicted_column = 'prediction'
        # Generate box plots for numeric columns
        plots_info = generate_box_plots(global_df, predicted_column)
        # Render EDA page with plots_info
        return render_template('eda.html', plots_info=plots_info)
    else:
        return "No data available for EDA."
    
@app.route('/')
def home():
    return render_template('main.html')
    

if __name__ == '__main__':
    app.run(debug=True,port=8000)