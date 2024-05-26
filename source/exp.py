
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import data, get_features, get_gender
from proxyMute import get_expl, ours_, iterative_analysis_
from training import blackbox_classifier


if __name__ == "__main__":
    print("*********** CLASSIFICATION EXPERIMENTS ********")
    plt.style.use('seaborn-deep')
    for i in range(1):
        print(f"Experiment started for fold {i}")
        profiles = data['fold_'+str(i)]['profiles']
        biased_labels = data['fold_'+str(i)]['biased_labels']
        blind_labels = data['fold_' + str(i)]['blind_labels']

        X = get_features(profiles['train']).copy(deep=True)
        X['label'] = biased_labels['train']
        X['gender'] = get_gender(profiles['train'])

        # STEP 2: train gender classifier and obtain feature importances per fold.
        test_preds, classifier = blackbox_classifier(get_features(profiles['train']), get_features(profiles['test']),
                                                 get_gender(profiles['train']), get_gender(profiles['test']))

        get_expl("val_"+str(i), classifier, profiles)

        # STEP 3: iteratively disable each feature cumulatively (ranked by importance decreasingly) and save results.
        iterative_analysis_(f"val_{i}", profiles, biased_labels['train'], biased_labels['test'], f"biased-test_{i}")

        # STEP 4: select the optimum (i) to disable and save fairness and performance measures.
        ours_(fold=i, folder="../faircvtest/", method=f"xgb_{i}")

