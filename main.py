import file_reader
import rocchio_classifier


def calc_accuracy(test_set, classifier):
    correct = 0.0
    total = len(test_set.keys())
    for key in test_set:
        real = test_set[key][-1]
        predicted = classifier.predict(test_set[key][0:-1])
        if real == predicted:
            correct += 1.0
    return correct / total


def calc_accuracy_cosine(test_set, classifier):
    correct = 0.0
    total = len(test_set.keys())
    for key in test_set:
        real = test_set[key][-1]
        predicted = classifier.predict_cosine(test_set[key][0:-1])
        if real == predicted:
            correct += 1.0
    return correct / total


if __name__ == '__main__':
    methods = ["Boolean", "tf", "tfidf", "tfidf with cosine similarity"]
    print('Accuracy results:')
    file_name = "./dataset/amazon_cells_labelled_full.txt"
    train_file_name = "./dataset/amazon_cells_labelled_train.txt"
    test_file_name = "./dataset/amazon_cells_labelled_test.txt"
    data = file_reader.FileReader(file_name)
    for method in methods:
        train_set, _ = data.build_set(method.lower(), train_file_name)
        test_set, _ = data.build_set(method.lower(), test_file_name)
        classifier = rocchio_classifier.RocchioClassifier(train_set)
        if method != "tfidf with cosine similarity":
            print(f"{method}:", '{:.3f}'.format(calc_accuracy(test_set, classifier)))
        else:
            print(f"{method}:", '{:.3f}'.format(calc_accuracy_cosine(test_set, classifier)))
