#import relevant libraries
from pickle import load
from keras.engine.saving import load_model
import numpy as np

model = load_model("train.h5")
scaler = load(open('scaler.pkl', 'rb'))
Xnew = np.array([[0.712715176, 0.130172962, 0.046944429, 0.091218199, 641, 36.95]])
print(Xnew)
xaa = scaler.transform(Xnew)
print("print FeaturesTest scalled: ")
print(xaa)
ynew = model.predict_classes(xaa)
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (xaa[i], ynew[i]))