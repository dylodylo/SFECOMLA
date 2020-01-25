import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error, mean_squared_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm

datasets = {
    0: ["../data/california_houses.csv", "median_house_value", "California Houses"],
    1: ["../data/newtrain.csv", "SalePrice", "Ames Houses"],
    2: ["../data/boston.csv", "sale_price", "Boston Houses"],
    3: ["../data/kingcounty.csv" , "price", "King County Houses"]
}

models = {
    0: "Linear Regression",
    1: "Tree Regression",
    2: "Random Forrest",
    3: "Gradient Boosting Regressor",
    4: "Bayesian Ridge Regression"
}

measures = {
    0: "MSE",
    1: "RMSE",
    2: "MAE",
    3: "R2",
    4: "MDAE"
}

def dataset_function(buttons_list, menu):
    data_checkboxes = buttons_list[0]
    algorithms_checkboxes = buttons_list[1]
    measures_checkboxes = buttons_list[2]
    ratio_checkboxes = buttons_list[3]

    datasets_list = []
    for idx, button in enumerate(data_checkboxes):
        if button:
            datasets_list.append(datasets[idx]) #dodajemy wybrane datasety do listy, żeby potem wyciągać ich ścieżki, nazwy i label column

    measures_list = ["Dataset", "Ratio", "Model"]

    for idx, button in enumerate(measures_checkboxes):
        if button:
            measures_list.append(measures[idx]) #to samo robimy z miarami

    row_list = []
    row_list.append(measures_list)

    for idx, dataset in enumerate(datasets_list): #działamy na każdym kolejnym datasecie
        menu.textBrowser.append(dataset[2])  # print nazwy datasetu
        df = pd.read_csv(dataset[0])

        for iratio, ratio in enumerate(ratio_checkboxes):
            if ratio:
                models_list = []
                train_ratio = iratio*0.1+0.5
                train_set, test_set = train_test_split(df, test_size=1-train_ratio, random_state=42)
                housing_prepared = train_set.drop(dataset[1], axis=1)  # usuwamy kolumnę z cenami bo je chcemy przewidywać
                housing_labels = train_set[dataset[1]].copy()  #tworzymy dataframe z samą wartością którą predykujemy

                print("Loading models")
                menu.textBrowser.append("Loading models")
                #sprawdamy użycie których modeli zadeklarował użytkownik
                if algorithms_checkboxes[0]:
                    lin_reg = LinearRegression()
                    lin_reg.fit(housing_prepared, housing_labels)
                    models_list.append([lin_reg, models[0]])
                    menu.textBrowser.append("Linear Regression loaded")
                if algorithms_checkboxes[1]:
                    tree_reg = DecisionTreeRegressor()
                    tree_reg.fit(housing_prepared, housing_labels)
                    models_list.append([tree_reg, models[1]])
                    menu.textBrowser.append("Decision Tree loaded")
                if algorithms_checkboxes[2]:
                    forest_reg = RandomForestRegressor()
                    forest_reg.fit(housing_prepared, housing_labels)
                    models_list.append([forest_reg, models[2]])
                    menu.textBrowser.append("Random Forest loaded")

                if algorithms_checkboxes[3]:
                    gbrt = GradientBoostingRegressor()
                    gbrt.fit(housing_prepared, housing_labels)
                    models_list.append([gbrt, models[3]])
                    menu.textBrowser.append("Gradient Boosting loaded")

                if algorithms_checkboxes[4]:
                    clf = BayesianRidge()
                    clf.fit(housing_prepared, housing_labels)
                    models_list.append([clf, models[4]])
                    menu.textBrowser.append("Bayesian Ridge loaded")

                menu.textBrowser.append('Models loaded')

                X_test_prepared = test_set.drop(dataset[1], axis=1) #odrzucamy kolumnę z cenami z zbioru testwoego
                y_test = test_set[dataset[1]].copy() #i dajemy je pod mienną y_test

                #czyszczenie pamięci
                test_set = []
                housing_prepared = []
                housing_labels = []

                #dla każdego modelu obliczamy miary
                for idx2, model in enumerate(models_list):
                    scores = [dataset[2], train_ratio, model[1]] #na początku wpisujemy co to za dataset i model
                    final_model = model[0]
                    final_predictions = final_model.predict(X_test_prepared)

                    #sprawdzamy użycie których miar zadeklarował użytkownik
                    if measures_checkboxes[0]:
                        final_mse = mean_squared_error(y_test, final_predictions)
                        scores.append(final_mse)

                    if measures_checkboxes[1]:
                        final_mse = mean_squared_error(y_test, final_predictions)
                        final_rmse = np.sqrt(final_mse)
                        scores.append(final_rmse)

                    if measures_checkboxes[2]:
                        final_mae = mean_absolute_error(y_test, final_predictions)
                        scores.append(final_mae)

                    if measures_checkboxes[3]:
                      # R Squared
                        final_r2score = r2_score(y_test, final_predictions)
                        scores.append(final_r2score)

                    if measures_checkboxes[4]:
                        final_mdae = median_absolute_error(y_test, final_predictions)
                        scores.append(final_mdae)

                    #czyszczenie pamięci
                    del model
                    del final_model
                    del final_predictions

                    row_list.append(scores) #dodajemy nazwę datasetu, nazwę modelu i jego miary do listy, którą potem użyjemy do przekazania danych


    with open('test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in row_list:
            writer.writerow(row)

    menu.pushButton.setEnabled(True)
    menu.pushButton_2.setEnabled(True)
    menu.textBrowser.append("Plotting...")