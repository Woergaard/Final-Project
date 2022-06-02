from modules.packages import *

def confusion_matrix_liar(df: pd.DataFrame(), col1: str, col2: str):    
    dummyarray = np.empty((2,2), dtype= int)
    dummyarray[:] = 0

    df = np.array(df.groupby([col1, col2]).size().unstack(fill_value=0))

    def func(a, b):
        x1, x2 = np.min((a.shape, b.shape), 0)
        c = a.copy()
        c[:x1, :x2] = b[:x1, :x2]
        return c

    array = func(dummyarray, df)

    return array


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')