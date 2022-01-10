import pickle

with open(r"C:\Users\krish\Downloads\dataset_samples\dataset_samples\cifar10\class_info.pkl", 'rb') as pickle_file:
    obj = pickle.load(pickle_file)

print(obj)