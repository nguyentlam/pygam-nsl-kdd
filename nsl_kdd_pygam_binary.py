from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from pygam import LogisticGAM, s, f
import matplotlib.pyplot as plt
import os

def parseNumber(s):
    try:
        return float(s)
    except ValueError:
        return s

data_train = np.loadtxt('./KDDTrain+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
data_test = np.loadtxt('./KDDTest+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
print('len(data_train)', len(data_train))
print('len(data_test)', len(data_test))

X_train_raw = data_train[:, 0:41]
y_train_raw = data_train[:, [41]]
print('X_train_raw[0:3]===========', X_train_raw[0:3])
print('y_train_raw[0:5]===========', y_train_raw[0:5])
print('=================')

X_test_raw = data_test[:, 0:41]
y_test_raw = data_test[:, [41]]
print('X_test_raw[0:3]===========', X_test_raw[0:3])
print('y_test_raw[0:3]===========', y_test_raw[0:3])
print('=================')

x_columns = np.array(list(range(41)))
print('x_columns', x_columns)
categorical_x_columns = np.array([1, 2, 3])
numberic_x_columns = np.delete(x_columns, categorical_x_columns)
print('numberic_x_columns', numberic_x_columns)
x_ct = ColumnTransformer(transformers = [("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_x_columns),
                                         ('normalize', Normalizer(norm='max'), numberic_x_columns)], remainder = 'passthrough')

x_ct.fit(X_train_raw)
X_train = x_ct.transform(X_train_raw)
X_train = X_train.astype('float')
print('X_train[0:3]', X_train[0:3])
print('len(X_train[0])', len(X_train[0]))

X_test = x_ct.transform(X_test_raw)
X_test = X_test.astype('float')
print('X_train[0:3]', X_train[0:3])
print('len(X_train[0])', len(X_train[0]))


def preprocess_label(y_data):
    print(y_data[0])
    return np.array(list(map(lambda x : 0 if x[0] == 'normal' else 1, y_data))).astype('float')

y_train = preprocess_label(y_train_raw)
#y_train = y_train.astype('float')
print('y_train[0:2]===', y_train[0:2])

y_test = preprocess_label(y_test_raw)
#y_test = y_test.astype('float')
print('y_test[0:2]===', y_test[0:2])

# Train a LogisticGAM classifier on the training set
gam = LogisticGAM(
    # terms='auto'
    s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) 
    + s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17) + s(18) + s(19) 
    + s(20) + s(21) + s(22) + s(23) + s(24) + s(25) + s(26) + s(27) + s(28) + s(29)
    + s(30) + s(31) + s(32) + s(33) + s(34) + s(35) + s(36) + s(37) + s(38) + s(39)
    + s(40) + s(41) + s(42) + s(43) + s(44) + s(45) + s(46) + s(47) + s(48) + s(49)
    + s(50) + s(51) + s(52) + s(53) + s(54) + s(55) + s(56) + s(57) + s(58) + s(59)
    + s(60) + s(61) + s(62) + s(63) + s(64) + s(65) + s(66) + s(67) + s(68) + s(69)
    + s(70) + s(71) + s(72) + s(73) + s(74) + s(75) + s(76) + s(77) + s(78) + s(79)
    + s(80) + s(81) + s(82) + s(83) + s(84) + s(85) + s(86) + s(87) + s(88) + s(89)
    + s(90) + s(91) + s(92) + s(93) + s(94) + s(95) + s(96) + s(97) + s(98) + s(99)
    + s(100) + s(101) + s(102) + s(103) + s(104) + s(105) + s(106) + s(107) + s(108) + s(109)
    + s(110) + s(111) + s(112) + s(113) + s(114) + s(115) + s(116) + s(117) + s(118) + s(119)
    + s(120) + s(121)
)

#gam.fit(X_train, y_train.ravel())
gam.gridsearch(X_train, y_train.ravel())

# Use the trained classifier to predict the classes of the test set
y_pred = gam.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

gam.summary()

fig, axs = plt.subplots(1, 122)
# titles = np.array(list(range(122)))

# for i, ax in enumerate(axs):
#     XX = gam.generate_X_grid(term=i)
#     pdep, confi = gam.partial_dependence(term=i, width=.95)

#     ax.plot(XX[:, i], pdep)
#     ax.plot(XX[:, i], confi, c='r', ls='--')
#     ax.set_title(titles[i])

# #plt.show()
# plt.savefig(os.path.basename(__file__).removesuffix(".py") + '.png')

# Create a 25x5 grid of subplots with larger subplots
fig, axes = plt.subplots(nrows=25, ncols=5, figsize=(50, 100))

# Loop through each subplot and plot the corresponding dataset
for i in range(25):
    for j in range(5):
        if i*5+j+1 <= 122:
            ax = axes[i, j]
            idx = i*5 + j
            XX = gam.generate_X_grid(term=idx)
            pdep, confi = gam.partial_dependence(term=idx, width=.95)

            ax.plot(XX[:, i], pdep)
            ax.plot(XX[:, i], confi, c='r', ls='--')
            ax.set_title(f"Subplot {idx+1}", fontsize=12)

# Adjust the spacing between subplots
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.4)

# Save the figure to a PDF file
plt.savefig("my_subplots.pdf", bbox_inches="tight")

# Show the figure (optional)
plt.show()