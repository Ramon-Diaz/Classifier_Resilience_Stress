# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from tqdm.auto import tqdm
from sys import stdout
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# %%
class RSIClassifier():

    def __init__(self, df):
        self.features_ = ['Electromyography', 'BloodVolume', 'Breathing', 'SkinConductance', 'CorporalTemperature']
        self.new_features_ = [element+'_diff' for element in self.features_]
        for i in self.features_:
            self.new_features_.append(i+'_diff2')
        
        self.results_ = {}
        self.resilience_index_ = {}
        self.scores_ = {}

        self.df_pc_ = df

        self.df_centroids_ = []
        self.distances_ = []

    def plot_freq(self, data_num, groups=[1,2,3,4,5]):
        values = self.df_pc_[data_num].values
        i = 1
        # plot each column
        plt.figure(figsize=(10, 8))
        for group in groups:
            plt.subplot(len(groups), 1, i)
            plt.plot(values[:, group])
            plt.title(self.df_pc_[data_num].columns[group], y=0.5, loc='right')
            i += 1
        plt.show()

        return None

    def cv_auc(self, data, classifier, cv):
        #print('    Training model: '+str(classifier))
        # Initialize variables
        auc_scores = []
        # Create a copy to avoid changes to original dataset
        db = data.copy()
        # define X and y
        y_ = db.Phase
        X = db.loc[:,self.features_+self.new_features_].values
        # Binarize the label
        classes = y_.unique()
        y = label_binarize(y_, classes=classes)
        # Begin the cross-validation method
        fold = 1
        #with tqdm(total=cv.get_n_splits(), leave=False, position=1) as pbar:
        for train_index, test_index in cv.split(X, y):
            #pbar.set_description('      Fold')
            # Split train-test
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Fit model
            y_pred = classifier.fit(X_train,y_train).predict(X_test)
            # Compute the AUC score
            score = roc_auc_score(y_test, y_pred)
            #score = 1-score if score<0.5 else score
            auc_scores.append(score)
            fold+=1
            #pbar.update(1)

        return auc_scores

    def get_models_results(self, subjects, models, cv):
        with tqdm(total=[True if 'phase5' in self.df_pc_[x].Phase.unique() else False for x in subjects].count(True), file=stdout) as pbart:
            for subject in subjects:
                if 'phase5' in self.df_pc_[subject].Phase.unique():
                    pbart.set_description('Subject')
                    #print('Subject: '+str(subject))
                    temp = self.df_pc_[subject].copy()
                    f_results_ = {}
                    test_df = (
                            ('phase1_vs_phase3',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase3')]),
                            ('phase1_vs_phase5',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase5')]),
                            ('phase3_vs_phase5',temp[(temp['Phase'] == 'phase3') | (temp['Phase'] == 'phase5')]),
                            ('phase1_vs_phase2',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase2')]),
                            ('phase1_vs_phase4',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase4')]),
                            ('phase3_vs_phase2',temp[(temp['Phase'] == 'phase3') | (temp['Phase'] == 'phase2')]),
                            ('phase3_vs_phase4',temp[(temp['Phase'] == 'phase3') | (temp['Phase'] == 'phase4')]),
                            ('phase5_vs_phase2',temp[(temp['Phase'] == 'phase5') | (temp['Phase'] == 'phase2')]),
                            ('phase5_vs_phase4',temp[(temp['Phase'] == 'phase5') | (temp['Phase'] == 'phase4')]) 
                        )
                    st = time.time()
                    counter = 1
                    for dataset in test_df:
                        #print('\n  Training data: '+dataset[0]+' ({}/{})\n'.format(counter,len(test_df)))
                        counter+=1
                        f_results_[dataset[0]] = {model_name: self.cv_auc(dataset[1], model, cv)  for model_name, model in models}
                    end = time.time()
                    #print('\n  Done training subject '+str(subject)+' in '+str(round((end-st)/60,2))+' minutes.\n')
                    self.results_[subject] = f_results_
                    pbart.update(1)
                else:
                    pass

        return self

    def boxplot_results(self, results_):
        model_names = [key for key in results_.get('phase1_vs_phase3').keys()]
        for key,values in results_.items():
            # print the graph
            _, ax = plt.subplots()
            ax.boxplot([model_value for model_value in values.values()])
            plt.title(key)
            ax.set_ylabel('AUC')
            ax.set_xticklabels(model_names,rotation=90)
            plt.show()

        return self

    def calculate_centroids(self):
        for subject in range(len(self.df_pc_)):
            self.df_centroids_.append(self.df_pc_[subject].loc[:,self.df_pc_[subject].columns !='Time'].groupby(by='Phase', as_index=False).mean())
        return self

    def euclidean_distance(self):
        distances_list = []
        for subject in range(len(self.df_centroids_)):
            distances_list.append([np.sqrt((self.df_centroids_[subject].Electromyography[0]-self.df_centroids_[subject].Electromyography[element])**2+\
                                            (self.df_centroids_[subject].BloodVolume[0]-self.df_centroids_[subject].BloodVolume[element])**2+\
                                            (self.df_centroids_[subject].Breathing[0]-self.df_centroids_[subject].Breathing[element])**2+\
                                            (self.df_centroids_[subject].SkinConductance[0]-self.df_centroids_[subject].SkinConductance[element])**2+\
                                            (self.df_centroids_[subject].CorporalTemperature[0]-self.df_centroids_[subject].CorporalTemperature[element])**2+\
                                            (self.df_centroids_[subject].Electromyography_diff[0]-self.df_centroids_[subject].Electromyography_diff[element])**2+\
                                            (self.df_centroids_[subject].BloodVolume_diff[0]-self.df_centroids_[subject].BloodVolume_diff[element])**2+\
                                            (self.df_centroids_[subject].Breathing_diff[0]-self.df_centroids_[subject].Breathing_diff[element])**2+\
                                            (self.df_centroids_[subject].SkinConductance_diff[0]-self.df_centroids_[subject].SkinConductance_diff[element])**2+\
                                            (self.df_centroids_[subject].CorporalTemperature_diff[0]-self.df_centroids_[subject].CorporalTemperature_diff[element])**2) for element in range(1, len(self.df_centroids_[subject]))])
        # This will get you the dataframe
        self.distances_ = pd.DataFrame(data=distances_list, columns=['Ph1-Ph2','Ph1-Ph3','Ph1-Ph4','Ph1-Ph5','Ph1-Ph6'])

        return self
    
    def calculate_resilience_index(self, results_):
        
        for subject_key in results_:
            resilience_index = []
            for key,values in results_.get(subject_key).items():
                # Get the max algorithm of the mean result of each fold
                resilience_index.append((key,np.max([np.mean(fold_results) for fold_results in values.values()])))
            # Append the resilience index
            self.resilience_index_[subject_key] = np.mean([resilience_index[j][1] for j in range(3,9)])-np.mean([resilience_index[i][1] for i in range(0,3)])
            # Save the complete dictionary
            self.scores_[subject_key] = resilience_index
        
        return self
    
    def results_to_dataframe(self, results_):
        temp = {(level1_key, level2_key, level3_key): values
                for level1_key, level2_dict in results_.items()
                for level2_key, level3_dict in level2_dict.items()
                for level3_key, values      in level3_dict.items()}
        temp_df = pd.DataFrame(temp).T
        temp_df.columns = ['fold_'+str(i+1) for i in range(len(temp_df.columns.values))]
        temp_df = temp_df.reset_index().rename(columns={'level_0':'Subject','level_1':'Phase_vs_phase','level_2':'model'})
        temp_df.to_csv('final_results.csv', index=False)

        return self
    
    def scores_to_dataframe(self, results_):
        temp = {(level1_key, level2_name): values
                for level1_key, level2_dict in results_.items()
                for level2_name, values in level2_dict}
        temp_df = pd.Series(temp).rename_axis(['Subject','Phase_vs_phase']).reset_index(name='Max_AUC_Average')
        temp_df.to_csv('final_scores.csv', index=False)

        return self

    def rsi_to_dataframe(self, resilience_index_):
        temp_df = pd.DataFrame(resilience_index_, index=['Classifier_RSI']).T
        temp_df = temp_df.reset_index().rename(columns={'index':'Subject'})
        temp_df.to_csv('final_rsi.csv', index=False)
        return self

    def export_data(self):
        self.results_to_dataframe(self.results_)
        self.scores_to_dataframe(self.scores_)
        self.rsi_to_dataframe(self.resilience_index_)
        
        return self
    
# %%
if __name__ == "__main__":
    print('Importing the data...')
    st = time.time()
    with open('subjects.data','rb') as data:
        df = pickle.load(data)
    end = time.time()
    print('Done importing in '+str(round(end-st,2))+' seconds.')

    model = RSIClassifier(df.copy())

    rs = 1
    model_list = [  
        ('KNNC',KNeighborsClassifier(5, n_jobs=-1)),
        ('LRC',LogisticRegression(n_jobs=-1,random_state=rs)),
        ('RFC',RandomForestClassifier(n_jobs=-1,random_state=rs)),
        ('QDA',QuadraticDiscriminantAnalysis()),
        ('GNBC',GaussianNB()),
        ('LDA',LinearDiscriminantAnalysis())
    ]
    subjects = range(0,len(model.df_pc_))
    cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=rs)
    model.get_models_results(subjects, model_list, cv)
    model.calculate_resilience_index(model.results_)
    print('Exporting the data...')
    model.export_data()
    with open('RSIClassifier.obj','wb') as data:
        pickle.dump(model, data)
    print('DONE.')
# %%
