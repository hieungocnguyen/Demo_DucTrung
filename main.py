from flask import Flask, request, render_template, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

# Flask constructor
app = Flask(__name__)

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

# A decorator used to tell the application
# which URL is associated function
@app.route("/login", methods=["POST"])
def login():
    path = 'Consumer_Complaints.csv'
    dataset = pd.read_csv(path)
    col = ['Product', 'Consumer Complaint']
    dataset = dataset[col]

    # Tải data và xóa những dòng bị thiếu
    dataset.dropna(subset=["Consumer Complaint"], inplace=True)
    dataset.columns = ['Product', 'ConsumerComplaint']  # Đổi tên cột
    dataset['category_id'] = dataset['Product'].factorize()[0]

    """## Xử dụng TF IDF Vectorizer trên data dạng text"""

    # Remove stop words
    stopwords = ["________________________________________from",
                 "________________",
                 "____",
                 "00",
                 "xxxx",
                 "xx", "a", "about",
                 "above", "after", "again",
                 "against", "all", "am",
                 "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below",
                 "between", "both", "but", "by", "could", "did", "do", "does",
                 "doing", "down", "during", "each", "few", "for", "from", "further",
                 "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her",
                 "here", "here's", "hers", "herself", "him", "himself", "his", "how",
                 "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
                 "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself",
                 "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
                 "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
                 "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
                 "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very",
                 "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's",
                 "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
                 "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

    for word in stopwords:
        dataset['ConsumerComplaint'] = dataset['ConsumerComplaint'].replace(to_replace=r'\b%s\b' % word, value="",
                                                                            regex=True)

    dataset['ConsumerComplaint'] = dataset['ConsumerComplaint'].str.replace('\d+', '')

    # Convert to lowercase
    dataset['ConsumerComplaint'] = dataset['ConsumerComplaint'].str.lower()

    x = dataset["ConsumerComplaint"]

    cv2 = TfidfVectorizer(stop_words='english', max_features=20000)
    X_traincv = cv2.fit_transform(x)
    X_traincv = normalize(X_traincv)

    a = pd.DataFrame(X_traincv.toarray(), columns=cv2.get_feature_names())

    """## Train model để tìm 5 cluster"""

    model = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1)
    model.fit(X_traincv)

    """## Top terms per cluster"""

    centroids = model.cluster_centers_
    labels = model.labels_

    order_centroids = centroids.argsort()[:, ::-1]
    terms = cv2.get_feature_names()
    for i in range(5):
        top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
        print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))

    """## Thực hiện predictions và hiện kết quả"""

    from sklearn.decomposition import PCA

    # First: for every document we get its corresponding cluster
    clusters = model.predict(X_traincv)

    # We train the PCA on the dense version of the tf-idf.
    pca = PCA(n_components=2)
    two_dim = pca.fit_transform(X_traincv.todense())

    scatter_x = two_dim[:, 0]  # first principle component
    scatter_y = two_dim[:, 1]  # second principle component

    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use('ggplot')

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)

    # color map for NUMBER_OF_CLUSTERS we have
    cmap = {0: 'orange', 1: 'blue', 2: 'red', 3: 'yellow', 4: 'black'}

    # group by clusters and scatter plot every cluster
    # with a colour and a label

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    terms = cv2.get_feature_names()
    for i in range(5):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

    predicted = model.predict(cv2.transform([request.form["question"]]))
    # print(predicted)
    a = predicted[0]

    def numbers_to_strings(a):
        switcher = {
            0: "xxxx card account debt bank",
            1: "credit report xxxx information reporting",
            2: "xxxx account credit payment bank",
            3: "xxxx loan payment mortgage payments",
            4: "xx xxxx account credit loan",
        }
        return switcher.get(a, "nothing")
    return str(numbers_to_strings(a))



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)