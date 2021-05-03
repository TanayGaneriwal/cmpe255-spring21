#In this assignment, you will be building a SVM classifier to label famous people's images.
#Omports
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Loading the Data
def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    print(faces.images.shape)
    return faces
faces = load_data()

fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],xlabel=faces.target_names[faces.target[i]])

#Randomized PCA
pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)


#Splitting
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)
                                                
#Grid Search CV
param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, ytrain)
print(grid.best_params_)
model = grid.best_estimator_
yfit = model.predict(Xtest)

#Classification Report
print("\nClassification Report\n")
print(classification_report(ytest, yfit, target_names=faces.target_names))

#Draw a 4x6 subplots of images using names as label with color black for correct instances and red for incorrect instances.
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);
plt.show()

#Draw a confusion matrix between features in a heatmap with X-axis of 'Actual' and Y-axis of 'Predicted'.
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=faces.target_names,yticklabels=faces.target_names)
plt.xlabel('Actual')
plt.ylabel('Predicted');
plt.title("Confusion Matrix\n")
plt.show()

