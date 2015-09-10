from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import *
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
import nltk
#nltk.download('all')
from nltk.stem.porter import PorterStemmer
import string
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers

token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# text preprocessing. Performs tokenizing and stemming.
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

# Read input csv file(s) into lists
def read_csv(file_path):
    
    data = np.loadtxt(file_path, dtype='str', delimiter="\t")

    Y = data[:,0]
    X_text = data[:,1]

    X_train, X_test, Y_train, Y_test = train_test_split(X_text, Y, test_size=0.33, random_state=42)

    return (X_train, X_test, Y_train, Y_test)

# Run 6-fold cross validation and find best estimator while iterating through multiple
#  combinations of parameters. Uses sklearn's TfidfVectorizer and GridSearchCV

def best_estimate(X_text, Y):

    # parameters for tfidfVectroizer and LinearSVC
    #params = {"svd__n_components": [100, 200, 300],
    params = {"pca__n_components": [100, 300, 500],
            "tfidf__ngram_range": [(1, 2)],
            "tfidf__max_df": [0.75],
            "tfidf__max_features": [None],
            "svc__C": [0.01, 0.1, 1, 10, 100]}

 
    clf = Pipeline([("tfidf", TfidfVectorizer(strip_accents='unicode', tokenizer=tokenize, sublinear_tf=True)),
                    ("pca", TruncatedSVD(tol=0.01)),
                    ("svc", LinearSVC())])

    print "\n\n Finding best estimator \n"

    # setting refit=False so that custom model can be trained
    gs = GridSearchCV(clf, params, verbose=2, n_jobs=-1, cv=6, refit=True)
    gs.fit(X_text, Y)

    print"\n\nBest estimator:\n"

    best_parameters = gs.best_estimator_.get_params()

    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return gs

def plot_scores(gs):

    a=[]
    b=[]
    c=[]
    for params, mean_score, scores in gs.grid_scores_:
        if(params["pca__n_components"] == 100):
            a.append((params["svc__C"], mean_score))
        elif(params["pca__n_components"] == 300):
            b.append((params["svc__C"], mean_score))
        else:
            c.append((params["svc__C"], mean_score))

    m,n = np.array(a)[:,0], np.array(a)[:,1]
    plt.figure(0)
    plt.plot(m, n, 'r')
    plt.xlabel('C')
    plt.ylabel('mean_score')
    plt.title('SVD truncation = 100')

    m,n = np.array(b)[:,0], np.array(b)[:,1]
    plt.figure(1)
    plt.plot(m, n, 'b')
    plt.xlabel('C')
    plt.ylabel('mean_score')
    plt.title('SVD truncation = 300')

    m,n = np.array(b)[:,0], np.array(b)[:,1]
    plt.figure(2)
    plt.plot(m, n, 'c')
    plt.xlabel('C')
    plt.ylabel('mean_score')
    plt.title('SVD truncation = 500')
    plt.show()



#classify test data
def classify_test_data(gs):
    
    while(1):
        strn = raw_input()

        print strn, " -> ", gs.predict([strn])

        if(strn == "exit"):
            break
    return


# transform text data into feature vectors using best estimator parameters
# and convert Y labels to -1 and 1 for training custom classifier
# input : gridsearchcv data and outputs X and Y i.e. numerical feature values and binary integer labels

def get_transformed_data(gs, X_text, Y):

    best_parameters = gs.best_estimator_.get_params()

    vect = TfidfVectorizer(strip_accents='unicode', tokenizer=tokenize,
            ngram_range=best_parameters["tfidf__ngram_range"],
            max_df=best_parameters["tfidf__max_df"],
            max_features=best_parameters["tfidf__max_features"])
    X = vect.fit_transform(X_text)

    lb = preprocessing.LabelBinarizer(neg_label=-1)
    Y = lb.fit_transform(Y)
    C = best_parameters["svc__C"]
    return (X, Y, C, vect)


# implement custom SVM using quadratic programming offered by cvxopt library
# contains fit and predict functions

class MY_SVM(object):

    def __init__(self, C):
        self.C = C
        self.C = float(self.C)

    def fit(self, X, Y):

        n_samples, n_features = X.shape

        # gram matrix
        K = X.dot(X.T)
        P = cvxopt.matrix(np.outer(Y, Y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(Y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))

        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = Y[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)           
            y_predict[i] = s
        return y_predict + self,
                    
    def predict(self, X):
        return np.sign(self.project(X))


def main():
   
    X_text = []

    file_path = "movie-reviews-sentiment.tsv"

    X_train, X_test, Y_train, Y_test = read_csv(file_path)

    gs = best_estimate(X_train, Y_train)

    # X, Y, C, vect = get_transformed_data(gs, X_text, Y) # get transformed data using best estimator paramters for use with custom classifier

    # reduce feature dimension
    # svd = TruncatedSVD(n_components=5, algorithm="arpack", tol = 1e-3)
    # X = svd.fit_transform(X)

    #classify_test_data(X, Y)
    # svm = MY_SVM(C)

    # print "\n\nUsing custom svm to train with complete train data\n\n"
    # x1,y1 = X.shape
    # print "x1 = ", x1, "x2 = ", y1
    # svm.fit(X, Y)
    # print "\n\nCompleted training data\n\n"

    #classify_test_data(gs, X_test, Y_test)

    print "\n\nTesting model..."
    print "\n\n Test data accuracy: ", gs.score(X_test, Y_test)

    plot_scores(gs)

if __name__ == '__main__':
    main()
