import numpy as np
import pandas as pd
import pydicom
import os
from PIL import Image

def change_size(in_file, width, length):
    if in_file[-3:] == "dcm":
        dataset = pydicom.dcmread(in_file)
        img = np.copy(dataset.pixel_array)
        img = img[0:width, 0 : length]
        #now need to save
        np.save(in_file[:-3] + "npy", img)
    elif in_file[-3:] == "png":
        img = np.array(Image.open(in_file))
        img = img[0:width, 0: length]
        np.save(in_file[:-3] + "npy", img)

def create_numpy_files(beg_path):
   csv_train_path = "/Users/alan/Documents/bdrad/mass_case_description_train_set.csv"
   csv_test_path = "/Users/alan/Documents/bdrad/mass_case_description_test_set.csv"
   df_train = pd.read_csv(csv_train_path)
   df_test = pd.read_csv(csv_test_path)
   
   for subdir, dirs, files in os.walk(beg_path):
      for file in files:
         filepath = subdir + os.sep + file
         
         if filepath.endswith(".dcm"):
            suffix = "test" if "Test" in filepath else "train"
            change_size(filepath, 229, 229)
            dataset = pydicom.dcmread(filepath)
            print(dataset.PatientName)
            combine(filepath[:-3] + "npy", get_breast_density(df_train,str(dataset.PatientName)))
            if get_breast_density(df_train, str(dataset.PatientName)) is None:
               print(str(dataset.PatientName) + "is not in training set")
               combine(filepath[:-3] + "npy", get_breast_density(df_test, str(dataset.PatientName)))

#Helper functions
def combine(img_path, breast_density):
   img = np.load(img_path)
   np.savez(img_path[:-3] + "npz", x = img, y = np.array([breast_density]))
   print("Saving to " + str(img_path[:-3] + "npz"))

#This returns the breast desntiy associated with a specific p_i
#returns None if there was no file found
def get_breast_density(df, patient_name):
   for i in range(len(df)):
      if patient_name in df.iloc[i][11]:
         return 1 if df.iloc[i][1] > 2 else 0
         
def create_numpyz_files(beg_path):
   for subdir, dirs, files in os.walk(beg_path):
      for file in files:
         filepath = subdir + os.sep + file
         
         if filepath.endswith(".npz"):
            combine(filepath, get_breast_density(..., ...),filepath[:-3] + ".npz")



