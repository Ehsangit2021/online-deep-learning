from math import gamma
from cv2 import fastNlMeansDenoising
from numpy.core.shape_base import vstack
import streamlit as st
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
from streamlit.type_util import Key
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import json
import seaborn as sns
import plotly.express as px
from streamlit_lottie import st_lottie
from sklearn.cluster import DBSCAN
from sklearn import preprocessing, datasets
from sklearn.neighbors import DistanceMetric
from utils import tri_center1, usingDeepModels, second_level_augmentation_deep, clustering_4_non_dominated_members
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
from streamlit_juxtapose import juxtapose
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os, datetime
# from stqdm import stqdm
# importing module


# import logging
 
# # Create and configure logger
# logging.basicConfig(filename='Logs.log',
#                     format='%(levelname)s %(asctime)s - %(message)s',
#                     filemode='a')
 
# # Creating an object
# logger = logging.getLogger()
 
# # Setting the threshold of logger to DEBUG
# logger.setLevel(logging.DEBUG)
 
# # Test messages
# logger.debug("Harmless debug Message")
# logger.info("Just an information")
# logger.warning("Its a Warning")
# logger.error("Did you try to divide by zero")
# logger.critical("Internet is down")
# # logging.shutdown()
#*******************************************************************************************************

@st.cache(suppress_st_warning=True)
def reportPandas(data, target):
    illustration = pd.DataFrame(np.copy(data))
    illustration['target'] = target
    illustration.describe()
    pr = illustration.profile_report()
    st_profile_report(pr)
    st.write("Entire input data:")
    st.table(illustration)


# stqdm.pandas()


clf = AdaBoostClassifier(n_estimators=100, random_state=0)

#*******************************************************************************************************
st.set_page_config(
     page_title="Deep Learning Enhancement with the Clustering using data-augmentation",
     page_icon="icon.png",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'http://aval-praxeology.com',
         'Report a bug': "http://akbazar.profcms.um.ac.ir/",
         'About': "# This app is based on the research take placed in a master work! akbarzadeh@ieee.org"
     }
 )
st.markdown("""
                <style> #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
            """, unsafe_allow_html=True)


with open("anim.json", "r") as f:
    data = json.load(f)


st.title("Deep Learning Enhancement with the Clustering using data-augmentation")
st.header("Shahrood University of Technology")
with open('Log.txt', 'a') as f:
    f.write("***************************************************")
    f.write(str(datetime.datetime.now()))
    f.write("*******************************************************\n")
# logger.info("**********************************************************************************************************")

row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((.1, 0.4, .1, 1.5, .1))

with row2_1:
    st_lottie(data)
    
    dataset_option = st.selectbox(
        'Please select the prefered data-set!',
        ( 'iris', 'wine', 'diabetes', 'files', 'linnerud', 'sample_image', \
         'sample_images', 'svmlight_file', 'svmlight_files'), index=0) # 'boston', 
        #  'diabetes', 'files', 'linnerud', 'sample_image', 'digits', 'breast_cancer'
        # 'sample_images', 'svmlight_file', 'svmlight_files'),
         

with row2_2:
    st.write("""
                You may upload your 2D data-base to see the performance. \
                Please see the sample plots.""")
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
    st.write("The first two columns considered as input and the last column as target!")
    show_data1 = st.checkbox("See the Test and Train data?", value=False)
    show_data2 = st.checkbox("See the raw data and corresponding analysis?", value=False)

#*******************************************************************************************************
if uploaded_files:
    dataset_option = uploaded_files.name
    with row2_1:
        database_name = 'You selected: <font style="font-family:sans-serif; color:Blue; font-size: 20px;">' \
                        + dataset_option + '</font>'
        st.markdown(database_name, unsafe_allow_html=True)
        # st.write('You selected:', dataset_option)
    dataFetched = pd.read_csv(uploaded_files)
    features = dataFetched.columns
    target = np.array(dataFetched[dataFetched.columns[-1:]].values.squeeze())
    dataFetched = dataFetched[dataFetched.columns[:-1]]
    data = dataFetched
else:
    with row2_1:
        database_name = 'You selected: <p style="font-family:sans-serif; color:Blue; font-size: 20px;">' \
                        + dataset_option + '</p>'
        st.markdown(database_name, unsafe_allow_html=True)
        # st.write('You selected:', dataset_option)
    exec("dataFetched=datasets.load_" +  dataset_option +"()")
    # with numpy.errstate(divide='ignore'):
    target = dataFetched.target
    data = dataFetched.data
    features = dataFetched.feature_names

with open('Log.txt', 'a') as f:
    f.write("Data-sets: " + dataset_option + "\n")
    f.write("Data-sets size: " + str(np.shape(data)[0]) + "," + str(np.shape(data)[1]) + "\n")
# logger.info("Data-sets:" + database_name)

min_max_scaler = preprocessing.MinMaxScaler()
data = pd.DataFrame(min_max_scaler.fit_transform(data))
st.write("size of input data is: " + str(data.shape))

#*******************************************************************************************************

accuracy, X_test, Y_test, X_train, Y_train = usingDeepModels(data, target)

row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((.1, 0.5, .1, 0.5, .1))
if show_data1:
    with row1_1:
        st.write("X_test is: ", X_test)
    
    with row1_2:
        st.write("Y_test is: ")
        st.write(Y_test)

st.title('Deep Dense model Accuracy is : %.2f' % (accuracy*100))
with open('Log.txt', 'a') as f:
    f.write('Deep Dense model Accuracy is : %.2f \n' % (accuracy*100))
# logger.info('Deep Dense model Accuracy is : %.2f' % (accuracy*100))

st.title('Adaboost Model:')
row11_space1, row11_1, row11_space2, row11_2, row11_space3 = st.columns((.05, 0.5, .25, 0.5, .05))


with row11_1:
    #define an ensemble model ADABOOST
    clf.fit(X_train, Y_train)
    targetHat = clf.predict(X_test)
    t = confusion_matrix(Y_test, targetHat)
    sns.heatmap(t, annot=True)
    st.pyplot(plt, use_container_width=True)
    matrix = classification_report(Y_test,targetHat,labels=np.unique(target))
    st.write('**Classification report for ADABoost** : \n',matrix)
    st.write("The accuracy for ADABoost model is: %", 100*accuracy_score(Y_test, targetHat))
    with open('Log.txt', 'a') as f:
        f.write("The accuracy for ADABoost model is: %"+str(100*accuracy_score(Y_test, targetHat)) + "\n")
    # logger.info("The accuracy for ADABoost model is: %**", 100*accuracy_score(Y_test, targetHat))

with row11_space2:
    axes_1 = st.number_input("Horizontal axis", min_value=1, max_value=np.shape(data)[1] - 1,
                             value=1, step=1, help="Choose the horizontal axis of diagram!")
    axes_2 = st.number_input("Vertical axis", min_value=1, max_value=np.shape(data)[1] - 1,
                             value=2, step=1, help="Choose the vertical axis of diagram!")

with row11_2:
    plt.clf()
    plt.scatter(data.iloc[:,axes_1],data.iloc[:,axes_2], c=target)
    plt.title("Data Number: " + str(np.shape(data)[0]))
    plt.xlabel(features[axes_1])
    plt.ylabel(features[axes_2])
    st.pyplot(plt, use_container_width=True)

#*******************************************************************************************************
#*******************************************************************************************************
#*******************************************************************************************************
#*******************************************************************************************************
#*******************************************************************************************************
#*******************************************************************************************************
#*******************************************************************************************************
#*******************************************************************************************************
#*******************************************************************************************************
#*******************************************************************************************************

# pureData, pureTarget = clustering_4_non_dominated_members(X_train, Y_train, hull)
pureData, pureTarget = X_train, Y_train

augmented_data = None
augmented_target = []
class_no = np.unique(pureTarget)
for class_label in range(len(class_no)):
    # class_current_member = np.where(pureTarget==class_label)

    hull = ConvexHull(pureData.iloc[pureTarget==class_label,:])

    ddata = pureData.iloc[pureTarget==class_label,:].iloc[hull.vertices,:]
    ttarget = pureTarget[pureTarget==class_label][hull.vertices]

    candidate, tttarget = second_level_augmentation_deep(ddata, ttarget)
    if augmented_data is None:
        augmented_data = candidate
        augmented_target = tttarget
    else:
        augmented_data = vstack((augmented_data,candidate))
        augmented_target = np.hstack((augmented_target,tttarget))
# st.write("The convex vertices are as follows:")
# st.table(hull.vertices)
# st.write('Centers are as follows: ')
# st.table(centers)

#*******************************************************************************************************

st.title('Augmented data:')
st.write('number of Augmented data:'+ str(np.shape(augmented_data)[0]))
with open('Log.txt', 'a') as f:
    f.write('number of Augmented data:'+ str(np.shape(augmented_data)[0]) + "\n")
# logger.info('number of Augmented data:'+ str(np.shape(augmented_data)[0]))

augmented_data_4_showing = augmented_data
augmented_target_4_shoing = augmented_target

augmented_data = np.vstack((augmented_data, X_train))
augmented_target = np.hstack((augmented_target, Y_train))


accuracy, X_test, Y_test, X_train, Y_train = usingDeepModels(augmented_data, augmented_target, X_test, Y_test, EPOC=100)
st.title('Deep Dense model Accuracy is : %.2f' % (accuracy*100))
with open('Log.txt', 'a') as f:
    f.write('Deep Dense model Accuracy is : %.2f \n' % (accuracy*100))
# logger.info('Deep Dense model Accuracy is : %.2f' % (accuracy*100))

row11_space1, row11_1, row11_space2, row11_2, row11_space3 = st.columns((.05, 0.5, .25, 0.5, .05))
with row11_1:
    #define an ensemble model ADABOOST
    clf.fit(augmented_data, augmented_target)
    targetHat = clf.predict(X_test)
    t = confusion_matrix(Y_test, targetHat)
    sns.heatmap(t, annot=True)
    st.pyplot(plt, use_container_width=True)
    matrix = classification_report(Y_test,targetHat,labels=np.unique(target))
    st.write('**Classification report for ADABoost** : \n',matrix)
    st.write("**The accuracy for ADABoost model is: %**", 100*accuracy_score(Y_test, targetHat))
    with open('Log.txt', 'a') as f:
        f.write("The accuracy for ADABoost model is: %"+ str(100*accuracy_score(Y_test, targetHat)) + "\n")
    # logger.info("The accuracy for ADABoost model is: %**", 100*accuracy_score(Y_test, targetHat))
with open('Log.txt', 'a') as f:
    f.write("************************************************************************************************************************************\n")
# logger.info("**********************************************************************************************************")

# with row11_space2:
#     axes_1 = st.number_input("Horizontal axis", min_value=1, max_value=np.shape(data)[1] - 1,
#                              value=1, step=1, help="Choose the horizontal axis of diagram!", key=2)
#     axes_2 = st.number_input("Vertical axis", min_value=1, max_value=np.shape(data)[1] - 1,
#                              value=2, step=1, help="Choose the vertical axis of diagram!", key=2)

with row11_2:
    plt.clf()
    plt.scatter(data.iloc[:,axes_1],data.iloc[:,axes_2], c=target)
    plt.scatter(pd.DataFrame(augmented_data_4_showing).iloc[:,axes_1],pd.DataFrame(augmented_data_4_showing).iloc[:,axes_2], c='red')
    plt.title("Data Number: " + str(np.shape(augmented_data)[0]) + \
              "\n Red points are augmented data")
    plt.xlabel(features[axes_1])
    plt.ylabel(features[axes_2])
    st.pyplot(plt, use_container_width=True)


    
row11_space11, row11_11, row11_space22, row11_22, row11_space44 = st.columns((.05, 0.5, .25, 0.5, .05))

with row11_2:
    plt.clf()
    plt.scatter(augmented_data[:,axes_1],augmented_data[:,axes_2], c=augmented_target)
    plt.title("Data Number: " + str(np.shape(augmented_data)[0]) + \
              "\n Red points are augmented data")
    plt.xlabel(features[axes_1])
    plt.ylabel(features[axes_2])
    st.pyplot(plt, use_container_width=True)