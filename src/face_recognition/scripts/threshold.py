def set_proba_thresh(model, vectorizer, validation_set, test_set, label_column, criterion, column):
    fpr = {}
    tpr = {}
    thr = {}
    features = vectorizer.transform(validation_set[column].values)
    for idx, label in enumerate(model.classes_):
        binary_labels = np.array([label_validation == label
                                  for label_validation in validation_set[label_column].values])
        fpr[idx], tpr[idx], thr[idx] = roc_curve(binary_labels, model.predict_proba(features)[:, idx]) # ++++

    plt.figure(figsize=(8,8))
    for idx, label in enumerate(model.classes_):
        roc_auc = auc(fpr[idx], tpr[idx])
        plt.plot(fpr[idx], tpr[idx], linewidth=2, label=label + ' AUC = ' + '{0:.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ROC')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    print('\n\033[1mVALIDATION RESULTS\n\033[0m')
    validation_result = []
    for lb_idx in thr:
        for idx, value in enumerate(fpr[lb_idx]):
            if value >= criterion:
                validation_result.append((model.classes_[lb_idx],
                                          fpr[lb_idx][idx], tpr[lb_idx][idx], thr[lb_idx][idx]))
