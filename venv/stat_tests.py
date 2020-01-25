import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import scipy.stats as ss
import statsmodels.api as sa
import scikit_posthocs as spc
import Orange
import matplotlib.pyplot as plt
import matplotlib
import statistics


def read_data(path):
    data = pd.read_csv(path, sep=',', header=0)
    return data

def FTest_5(alg1, alg2, alg3, alg4, alg5):
    return stats.friedmanchisquare(alg1, alg2, alg3, alg4, alg5)

def FTest_4(alg1, alg2, alg3, alg4):
    return stats.friedmanchisquare(alg1, alg2, alg3, alg4)

def FTest_3(alg1, alg2, alg3):
    return stats.friedmanchisquare(alg1, alg2, alg3)

def NemTest(list_of_alg_val):
    return spc.posthoc_nemenyi(list_of_alg_val)

def NemFTest(df):
    return spc.posthoc_nemenyi_friedman(df)

def makeDF_5(models_names, alg1, alg2, alg3, alg4, alg5):
    d = {models_names[0]: np.array(alg1), models_names[1]: np.array(alg2),
         models_names[2]: np.array(alg3), models_names[3]: np.array(alg4),
         models_names[4]: np.array(alg5)}
    df = pd.DataFrame(data=d)
    return df

def makeDF_4(models_names, alg1, alg2, alg3, alg4):
    d = {models_names[0]: np.array(alg1), models_names[1]: np.array(alg2),
         models_names[2]: np.array(alg3), models_names[3]: np.array(alg4)}
    df = pd.DataFrame(data=d)
    return df

def makeDF_3(models_names, alg1, alg2, alg3):
    d = {models_names[0]: np.array(alg1), models_names[1]: np.array(alg2),
         models_names[2]: np.array(alg3)}
    df = pd.DataFrame(data=d)
    return df

def count_cd(measure_list, num_test):
    return Orange.evaluation.scoring.compute_CD(measure_list, num_test)

def graph_cd(average, cd, name):
    return Orange.evaluation.scoring.graph_ranks(average, average.index, cd, width=10, textspace=3, filename = name)

def make_rank(df):
    return df.rank(method = 'min', axis = 1 , ascending = False).mean()





def statistic_tests(path, algorithms, measures):

    RF = algorithms[0]
    TR = algorithms[1]
    LR = algorithms[2]
    GBR = algorithms[3]
    BRR = algorithms[4]
    M_MSE = measures[0]
    M_RMSE = measures[1]
    M_MAE = measures[2]
    M_R2 = measures[3]
    M_MDAE = measures[4]

    data = read_data(path)
    data = data.sort_values(by=['Model'])

    model_names = data['Model'].unique()
    number_of_models = len(model_names)

    number_of_measures = len(data.columns) - 3

    graph_names = []
    graph_count = 1

    ######################### 3 MODELE #####################

    if RF and TR and LR and not GBR and not BRR:
        if M_MSE:
            i = 0
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [LR_MSE, RF_MSE, TR_MSE]

            print('MSE: ', FTest_3(LR_MSE, RF_MSE, TR_MSE))
            MSE_DF = makeDF_3(model_names, LR_MSE, RF_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')


        if M_RMSE:
            i = 0
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [LR_RMSE, RF_RMSE, TR_RMSE]

            print('RMSE: ', FTest_3(LR_RMSE, RF_RMSE, TR_RMSE))
            RMSE_DF = makeDF_3(model_names, LR_RMSE, RF_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [LR_MAE, RF_MAE, TR_MAE]

            print('MAE: ', FTest_3(LR_MAE, RF_MAE, TR_MAE))
            MAE_DF = makeDF_3(model_names, LR_MAE, RF_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [LR_R2, RF_R2, TR_R2]

            print('R2: ', FTest_3(LR_R2, RF_R2, TR_R2))
            R2_DF = makeDF_3(model_names, LR_R2, RF_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [LR_MDAE, RF_MDAE, TR_MDAE]

            print('MDAE: ', FTest_3(LR_MDAE, RF_MDAE, TR_MDAE))
            MDAE_DF = makeDF_3(model_names, LR_MDAE, RF_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if RF and TR and GBR and not LR and not BRR:
        if M_MSE:
            i = 0
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [GBR_MSE, RF_MSE, TR_MSE]

            print('MSE: ', FTest_3(GBR_MSE, RF_MSE, TR_MSE))
            MSE_DF = makeDF_3(model_names, GBR_MSE, RF_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [GBR_RMSE, RF_RMSE, TR_RMSE]

            print('RMSE: ', FTest_3(GBR_RMSE, RF_RMSE, TR_RMSE))
            RMSE_DF = makeDF_3(model_names, GBR_RMSE, RF_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [GBR_MAE, RF_MAE, TR_MAE]

            print('MAE: ', FTest_3(GBR_MAE, RF_MAE, TR_MAE))
            MAE_DF = makeDF_3(model_names, GBR_MAE, RF_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [GBR_R2, RF_R2, TR_R2]

            print('R2: ', FTest_3(GBR_R2, RF_R2, TR_R2))
            R2_DF = makeDF_3(model_names, GBR_R2, RF_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [GBR_MDAE, RF_MDAE, TR_MDAE]

            print('MDAE: ', FTest_3(GBR_MDAE, RF_MDAE, TR_MDAE))
            MDAE_DF = makeDF_3(model_names, GBR_MDAE, RF_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if RF and TR and BRR and not GBR and not LR:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, RF_MSE, TR_MSE]

            print('MSE: ', FTest_3(BRR_MSE, RF_MSE, TR_MSE))
            MSE_DF = makeDF_3(model_names, BRR_MSE, RF_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, RF_RMSE, TR_RMSE]

            print('RMSE: ', FTest_3(BRR_RMSE, RF_RMSE, TR_RMSE))
            RMSE_DF = makeDF_3(model_names, BRR_RMSE, RF_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, RF_MAE, TR_MAE]

            print('MAE: ', FTest_3(BRR_MAE, RF_MAE, TR_MAE))
            MAE_DF = makeDF_3(model_names, BRR_MAE, RF_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')



        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, RF_R2, TR_R2]

            print('R2: ', FTest_3(BRR_R2, RF_R2, TR_R2))
            R2_DF = makeDF_3(model_names, BRR_R2, RF_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, RF_MDAE, TR_MDAE]

            print('MDAE: ', FTest_3(BRR_MDAE, RF_MDAE, TR_MDAE))
            MDAE_DF = makeDF_3(model_names, BRR_MDAE, RF_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if RF and LR and GBR and not BRR and not TR:
        if M_MSE:
            i = 0
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE

            MSE = [GBR_MSE, LR_MSE, RF_MSE]

            print('MSE: ', FTest_3(GBR_MSE, LR_MSE, RF_MSE))
            MSE_DF = makeDF_3(model_names, GBR_MSE, LR_MSE, RF_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [GBR_RMSE, LR_RMSE, RF_RMSE]

            print('RMSE: ', FTest_3(GBR_RMSE, LR_RMSE, RF_RMSE))
            RMSE_DF = makeDF_3(model_names, GBR_RMSE, LR_RMSE, RF_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE

            MAE = [GBR_MAE, LR_MAE, RF_MAE]

            print('MAE: ', FTest_3(GBR_MAE, LR_MAE, RF_MAE))
            MAE_DF = makeDF_3(model_names, GBR_MAE, LR_MAE, RF_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2

            R2 = [GBR_R2, LR_R2, RF_R2]

            print('R2: ', FTest_3(GBR_R2, LR_R2, RF_R2))
            R2_DF = makeDF_3(model_names, GBR_R2, LR_R2, RF_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [GBR_MDAE, LR_MDAE, RF_MDAE]

            print('MDAE: ', FTest_3(GBR_MDAE, LR_MDAE, RF_MDAE))
            MDAE_DF = makeDF_3(model_names, GBR_MDAE, LR_MDAE, RF_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if RF and LR and BRR and not GBR and not TR:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, LR_MSE, RF_MSE]

            print('MSE: ', FTest_3(BRR_MSE, LR_MSE, RF_MSE))
            MSE_DF = makeDF_3(model_names, BRR_MSE, LR_MSE, RF_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, LR_RMSE, RF_RMSE]

            print('RMSE: ', FTest_3(BRR_RMSE, LR_RMSE, RF_RMSE))
            RMSE_DF = makeDF_3(model_names, BRR_RMSE, LR_RMSE, RF_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, LR_MAE, RF_MAE]

            print('MAE: ', FTest_3(BRR_MAE, LR_MAE, RF_MAE))
            MAE_DF = makeDF_3(model_names, BRR_MAE, LR_MAE, RF_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, LR_R2, RF_R2]

            print('R2: ', FTest_3(BRR_R2, LR_R2, RF_R2))
            R2_DF = makeDF_3(model_names, BRR_R2, LR_R2, RF_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, LR_MDAE, RF_MDAE]

            print('MDAE: ', FTest_3(BRR_MDAE, LR_MDAE, RF_MDAE))
            MDAE_DF = makeDF_3(model_names, BRR_MDAE, LR_MDAE, RF_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')


    if TR and LR and GBR and not BRR and not RF:
        if M_MSE:
            i = 0
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [GBR_MSE, LR_MSE, TR_MSE]

            print('MSE: ', FTest_3(GBR_MSE, LR_MSE, TR_MSE))
            MSE_DF = makeDF_3(model_names, GBR_MSE, LR_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [GBR_RMSE, LR_RMSE, TR_RMSE]

            print('RMSE: ', FTest_3(GBR_RMSE, LR_RMSE, TR_RMSE))
            RMSE_DF = makeDF_3(model_names, GBR_RMSE, LR_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [GBR_MAE, LR_MAE, TR_MAE]

            print('MAE: ', FTest_3(GBR_MAE, LR_MAE, TR_MAE))
            MAE_DF = makeDF_3(model_names, GBR_MAE, LR_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [GBR_R2, LR_R2, TR_R2]

            print('R2: ', FTest_3(GBR_R2, LR_R2, TR_R2))
            R2_DF = makeDF_3(model_names, GBR_R2, LR_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [GBR_MDAE, LR_MDAE, TR_MDAE]

            print('MDAE: ', FTest_3(GBR_MDAE, LR_MDAE, TR_MDAE))
            MDAE_DF = makeDF_3(model_names, GBR_MDAE, LR_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if TR and LR and BRR and not GBR and not RF:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, LR_MSE, TR_MSE]

            print('MSE: ', FTest_3(BRR_MSE, LR_MSE, TR_MSE))
            MSE_DF = makeDF_3(model_names, BRR_MSE, LR_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, LR_RMSE, TR_RMSE]

            print('RMSE: ', FTest_3(BRR_RMSE, LR_RMSE, TR_RMSE))
            RMSE_DF = makeDF_3(model_names, BRR_RMSE, LR_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, LR_MAE, TR_MAE]

            print('MAE: ', FTest_3(BRR_MAE, LR_MAE, TR_MAE))
            MAE_DF = makeDF_3(model_names, BRR_MAE, LR_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, LR_R2, TR_R2]

            print('R2: ', FTest_3(BRR_R2, LR_R2, TR_R2))
            R2_DF = makeDF_3(model_names, BRR_R2, LR_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, LR_MDAE, TR_MDAE]

            print('MDAE: ', FTest_3(BRR_MDAE, LR_MDAE, TR_MDAE))
            MDAE_DF = makeDF_3(model_names, BRR_MDAE, LR_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if TR and GBR and BRR and not LR and not RF:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, GBR_MSE, TR_MSE]

            print('MSE: ', FTest_3(BRR_MSE, GBR_MSE, TR_MSE))
            MSE_DF = makeDF_3(model_names, BRR_MSE, GBR_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, GBR_RMSE, TR_RMSE]

            print('RMSE: ', FTest_3(BRR_RMSE, GBR_RMSE, TR_RMSE))
            RMSE_DF = makeDF_3(model_names, BRR_RMSE, GBR_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, GBR_MAE, TR_MAE]

            print('MAE: ', FTest_3(BRR_MAE, GBR_MAE, TR_MAE))
            MAE_DF = makeDF_3(model_names, BRR_MAE, GBR_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, GBR_R2, TR_R2]

            print('R2: ', FTest_3(BRR_R2, GBR_R2, TR_R2))
            R2_DF = makeDF_3(model_names, BRR_R2, GBR_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, GBR_MDAE, TR_MDAE]

            print('MDAE: ', FTest_3(BRR_MDAE, GBR_MDAE, TR_MDAE))
            MDAE_DF = makeDF_3(model_names, BRR_MDAE, GBR_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if LR and GBR and BRR and not RF and not TR:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, GBR_MSE, LR_MSE]

            print('MSE: ', FTest_3(BRR_MSE, GBR_MSE, LR_MSE))
            MSE_DF = makeDF_3(model_names, BRR_MSE, GBR_MSE, LR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, GBR_RMSE, LR_RMSE]

            print('RMSE: ', FTest_3(BRR_RMSE, GBR_RMSE, LR_RMSE))
            RMSE_DF = makeDF_3(model_names, BRR_RMSE, GBR_RMSE, LR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, GBR_MAE, LR_MAE]

            print('MAE: ', FTest_3(BRR_MAE, GBR_MAE, LR_MAE))
            MAE_DF = makeDF_3(model_names, BRR_MAE, GBR_MAE, LR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, GBR_R2, LR_R2]

            print('R2: ', FTest_3(BRR_R2, GBR_R2, LR_R2))
            R2_DF = makeDF_3(model_names, BRR_R2, GBR_R2, LR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, GBR_MDAE, LR_MDAE]

            print('MDAE: ', FTest_3(BRR_MDAE, GBR_MDAE, LR_MDAE))
            MDAE_DF = makeDF_3(model_names, BRR_MDAE, GBR_MDAE, LR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if RF and GBR and BRR and not LR and not TR:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, GBR_MSE, RF_MSE]

            print('MSE: ', FTest_3(BRR_MSE, GBR_MSE, RF_MSE))
            MSE_DF = makeDF_3(model_names, BRR_MSE, GBR_MSE, RF_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, GBR_RMSE, RF_RMSE]

            print('RMSE: ', FTest_3(BRR_RMSE, GBR_RMSE, RF_RMSE))
            RMSE_DF = makeDF_3(model_names, BRR_RMSE, GBR_RMSE, RF_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, GBR_MAE, RF_MAE]

            print('MAE: ', FTest_3(BRR_MAE, GBR_MAE, RF_MAE))
            MAE_DF = makeDF_3(model_names, BRR_MAE, GBR_MAE, RF_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, GBR_R2, RF_R2]

            print('R2: ', FTest_3(BRR_R2, GBR_R2, RF_R2))
            R2_DF = makeDF_3(model_names, BRR_R2, GBR_R2, RF_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, GBR_MDAE, RF_MDAE]

            print('MDAE: ', FTest_3(BRR_MDAE, GBR_MDAE, RF_MDAE))
            MDAE_DF = makeDF_3(model_names, BRR_MDAE, GBR_MDAE, RF_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    ##############################################################


    ######################### 4 MODELE ###########################

    if RF and TR and LR and BRR and not GBR:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, LR_MSE, RF_MSE, TR_MSE]

            print('MSE: ', FTest_4(BRR_MSE, LR_MSE, RF_MSE, TR_MSE))
            MSE_DF = makeDF_4(model_names, BRR_MSE, LR_MSE, RF_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE]

            print('RMSE: ', FTest_4(BRR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE))
            RMSE_DF = makeDF_4(model_names, BRR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')


        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE


            MAE = [BRR_MAE, LR_MAE, RF_MAE, TR_MAE]

            print('MAE: ', FTest_4(BRR_MAE, LR_MAE, RF_MAE, TR_MAE))
            MAE_DF = makeDF_4(model_names, BRR_MAE, LR_MAE, RF_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')


        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2


            R2 = [BRR_R2, LR_R2, RF_R2, TR_R2]

            print('R2: ', FTest_4(BRR_R2, LR_R2, RF_R2, TR_R2))
            R2_DF = makeDF_4(model_names, BRR_R2, LR_R2, RF_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE]

            print('MDAE: ', FTest_4(BRR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE))
            MDAE_DF = makeDF_4(model_names, BRR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if RF and TR and LR and GBR and not BRR:
        if M_MSE:
            i = 0
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [GBR_MSE, LR_MSE, RF_MSE, TR_MSE]

            print('MSE: ', FTest_4(GBR_MSE, LR_MSE, RF_MSE, TR_MSE))
            MSE_DF = makeDF_4(model_names, GBR_MSE, LR_MSE, RF_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')


        if M_RMSE:
            i = 0
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [GBR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE]

            print('RMSE: ', FTest_4(GBR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE))
            RMSE_DF = makeDF_4(model_names, GBR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')


        if M_MAE:
            i = 0
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [GBR_MAE, LR_MAE, RF_MAE, TR_MAE]

            print('MAE: ', FTest_4(GBR_MAE, LR_MAE, RF_MAE, TR_MAE))
            MAE_DF = makeDF_4(model_names, GBR_MAE, LR_MAE, RF_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [GBR_R2, LR_R2, RF_R2, TR_R2]

            print('R2: ', FTest_4(GBR_R2, LR_R2, RF_R2, TR_R2))
            R2_DF = makeDF_4(model_names, GBR_R2, LR_R2, RF_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')


        if M_MDAE:
            i = 0
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [GBR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE]

            print('MDAE: ', FTest_4(GBR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE))
            MDAE_DF = makeDF_4(model_names, GBR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if RF and TR and GBR and BRR and not LR:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, GBR_MSE, RF_MSE, TR_MSE]

            print('MSE: ', FTest_4(BRR_MSE, GBR_MSE, RF_MSE, TR_MSE))
            MSE_DF = makeDF_4(model_names, BRR_MSE, GBR_MSE, RF_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, GBR_RMSE, RF_RMSE, TR_RMSE]

            print('RMSE: ', FTest_4(BRR_RMSE, GBR_RMSE, RF_RMSE, TR_RMSE))
            RMSE_DF = makeDF_4(model_names, BRR_RMSE, GBR_RMSE, RF_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')


        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, GBR_MAE, RF_MAE, TR_MAE]

            print('MAE: ', FTest_4(BRR_MAE, GBR_MAE, RF_MAE, TR_MAE))
            MAE_DF = makeDF_4(model_names, BRR_MAE, GBR_MAE, RF_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')


        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, GBR_R2, RF_R2, TR_R2]

            print('R2: ', FTest_4(BRR_R2, GBR_R2, RF_R2, TR_R2))
            R2_DF = makeDF_4(model_names, BRR_R2, GBR_R2, RF_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')


        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, GBR_MDAE, RF_MDAE, TR_MDAE]

            print('MDAE: ', FTest_4(BRR_MDAE, GBR_MDAE, RF_MDAE, TR_MDAE))
            MDAE_DF = makeDF_4(model_names, BRR_MDAE, GBR_MDAE, RF_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if RF and LR and GBR and BRR and not TR:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, GBR_MSE, LR_MSE, RF_MSE]

            print('MSE: ', FTest_4(BRR_MSE, GBR_MSE, LR_MSE, RF_MSE))
            MSE_DF = makeDF_4(model_names, BRR_MSE, GBR_MSE, LR_MSE, RF_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')

        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE


            RMSE = [BRR_RMSE, GBR_RMSE, LR_RMSE, RF_RMSE]

            print('RMSE: ', FTest_4(BRR_RMSE, GBR_RMSE, LR_RMSE, RF_RMSE))
            RMSE_DF = makeDF_4(model_names, BRR_RMSE, GBR_RMSE, LR_RMSE, RF_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, GBR_MAE, LR_MAE, RF_MAE]

            print('MAE: ', FTest_4(BRR_MAE, GBR_MAE, LR_MAE, RF_MAE))
            MAE_DF = makeDF_4(model_names, BRR_MAE, GBR_MAE, LR_MAE, RF_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, GBR_R2, LR_R2, RF_R2]

            print('R2: ', FTest_4(BRR_R2, GBR_R2, LR_R2, RF_R2))
            R2_DF = makeDF_4(model_names, BRR_R2, GBR_R2, LR_R2, RF_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, GBR_MDAE, LR_MDAE, RF_MDAE]

            print('MDAE: ', FTest_4(BRR_MDAE, GBR_MDAE, LR_MDAE, RF_MDAE))
            MDAE_DF = makeDF_4(model_names, BRR_MDAE, GBR_MDAE, LR_MDAE, RF_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    if TR and LR and GBR and BRR and not RF:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, GBR_MSE, LR_MSE, TR_MSE]

            print('MSE: ', FTest_4(BRR_MSE, GBR_MSE, LR_MSE, TR_MSE))
            MSE_DF = makeDF_4(model_names, BRR_MSE, GBR_MSE, LR_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')


        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, GBR_RMSE, LR_RMSE, TR_RMSE]

            print('RMSE: ', FTest_4(BRR_RMSE, GBR_RMSE, LR_RMSE, TR_RMSE))
            RMSE_DF = makeDF_4(model_names, BRR_RMSE, GBR_RMSE, LR_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')


        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, GBR_MAE, LR_MAE, TR_MAE]

            print('MAE: ', FTest_4(BRR_MAE, GBR_MAE, LR_MAE, TR_MAE))
            MAE_DF = makeDF_4(model_names, BRR_MAE, GBR_MAE, LR_MAE, TR_MAE)
            cd_MAE = count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')


        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, GBR_R2, LR_R2, TR_R2]

            print('R2: ', FTest_4(BRR_R2, GBR_R2, LR_R2, TR_R2))
            R2_DF = makeDF_4(model_names, BRR_R2, GBR_R2, LR_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, GBR_MDAE, LR_MDAE, TR_MDAE]

            print('MDAE: ', FTest_4(BRR_MDAE, GBR_MDAE, LR_MDAE, TR_MDAE))
            MDAE_DF = makeDF_4(model_names, BRR_MDAE, GBR_MDAE, LR_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

    #############################################################



    ########################5 MODELI ###########################

    if RF and TR and LR and GBR and BRR:
        if M_MSE:
            i = 0
            BRR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            GBR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            LR_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            RF_MSE = data[data.Model == model_names[i]].MSE
            i = i + 1
            TR_MSE = data[data.Model == model_names[i]].MSE

            MSE = [BRR_MSE, GBR_MSE, LR_MSE, RF_MSE, TR_MSE]

            print('MSE: ', FTest_5(BRR_MSE, GBR_MSE, LR_MSE, RF_MSE, TR_MSE))
            MSE_DF = makeDF_5(model_names, BRR_MSE, GBR_MSE, LR_MSE, RF_MSE, TR_MSE)
            cd_MSE = count_cd(MSE, len(MSE[0]))
            nem_MSE = NemFTest(MSE_DF)
            print('CD_MSE: ', cd_MSE)
            print('MSE: ', NemFTest(MSE_DF))
            average_by_data_MSE = make_rank(MSE_DF)
            print('Average rank by data: ', average_by_data_MSE)
            average_by_PH_MSE = make_rank(nem_MSE)
            print('Average rank by Nemenyi test: ', average_by_PH_MSE)
            print()
            graph_cd(average_by_data_MSE, cd_MSE, 'MSE_by_data')
            graph_cd(average_by_PH_MSE, cd_MSE, 'MSE_by_test')

            graph_names.append('MSE_by_data')
            graph_names.append('MSE_by_test')


        if M_RMSE:
            i = 0
            BRR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            GBR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            LR_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            RF_RMSE = data[data.Model == model_names[i]].RMSE
            i = i + 1
            TR_RMSE = data[data.Model == model_names[i]].RMSE

            RMSE = [BRR_RMSE, GBR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE]

            print('RMSE: ', FTest_5(BRR_RMSE, GBR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE))
            RMSE_DF = makeDF_5(model_names, BRR_RMSE, GBR_RMSE, LR_RMSE, RF_RMSE, TR_RMSE)
            cd_RMSE = count_cd(RMSE, len(RMSE[0]))
            nem_RMSE = NemFTest(RMSE_DF)
            print('CD_RMSE: ', cd_RMSE)
            print('RMSE: ', NemFTest(RMSE_DF))
            average_by_data_RMSE = make_rank(RMSE_DF)
            print('Average rank by data: ', average_by_data_RMSE)
            average_by_PH_RMSE = make_rank(nem_RMSE)
            print('Average rank by Nemenyi test: ', average_by_PH_RMSE)
            print()
            graph_cd(average_by_data_RMSE, cd_RMSE, 'RMSE_by_data')
            graph_cd(average_by_PH_RMSE, cd_RMSE, 'RMSE_by_test')

            graph_names.append('RMSE_by_data')
            graph_names.append('RMSE_by_test')

        if M_R2:
            i = 0
            BRR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            GBR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            LR_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            RF_R2 = data[data.Model == model_names[i]].R2
            i = i + 1
            TR_R2 = data[data.Model == model_names[i]].R2

            R2 = [BRR_R2, GBR_R2, LR_R2, RF_R2, TR_R2]

            print('R2: ', FTest_5(BRR_R2, GBR_R2, LR_R2, RF_R2, TR_R2))
            R2_DF = makeDF_5(model_names, BRR_R2, GBR_R2, LR_R2, RF_R2, TR_R2)
            cd_R2 = count_cd(R2, len(R2[0]))
            nem_R2 = NemFTest(R2_DF)
            print('CD_R2: ', cd_R2)
            print('R2: ', NemFTest(R2_DF))
            average_by_data_R2 = make_rank(R2_DF)
            print('Average rank by data: ', average_by_data_R2)
            average_by_PH_R2 = make_rank(nem_R2)
            print('Average rank by Nemenyi test: ', average_by_PH_R2)
            print()
            graph_cd(average_by_data_R2, cd_R2, 'R2_by_data')
            graph_cd(average_by_PH_R2, cd_R2, 'R2_by_test')

            graph_names.append('R2_by_data')
            graph_names.append('R2_by_test')

        if M_MDAE:
            i = 0
            BRR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            GBR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            LR_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            RF_MDAE = data[data.Model == model_names[i]].MDAE
            i = i + 1
            TR_MDAE = data[data.Model == model_names[i]].MDAE

            MDAE = [BRR_MDAE, GBR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE]

            print('MDAE: ', FTest_5(BRR_MDAE, GBR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE))
            MDAE_DF = makeDF_5(model_names, BRR_MDAE, GBR_MDAE, LR_MDAE, RF_MDAE, TR_MDAE)
            cd_MDAE = count_cd(MDAE, len(MDAE[0]))
            nem_MDAE = NemFTest(MDAE_DF)
            print('CD_MDAE: ', cd_MDAE)
            print('MDAE: ', NemFTest(MDAE_DF))
            average_by_data_MDAE = make_rank(MDAE_DF)
            print('Average rank by data: ', average_by_data_MDAE)
            average_by_PH_MDAE = make_rank(nem_MDAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MDAE)
            print()
            graph_cd(average_by_data_MDAE, cd_MDAE, 'MDAE_by_data')
            graph_cd(average_by_PH_MDAE, cd_MDAE, 'MDAE_by_test')

            graph_names.append('MDAE_by_data')
            graph_names.append('MDAE_by_test')

        if M_MAE:
            i = 0
            BRR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            GBR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            LR_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            RF_MAE = data[data.Model == model_names[i]].MAE
            i = i + 1
            TR_MAE = data[data.Model == model_names[i]].MAE

            MAE = [BRR_MAE, GBR_MAE, LR_MAE, RF_MAE, TR_MAE]

            print('MAE: ',FTest_5(BRR_MAE, GBR_MAE, LR_MAE, RF_MAE, TR_MAE))
            MAE_DF = makeDF_5(model_names, BRR_MAE, GBR_MAE, LR_MAE, RF_MAE, TR_MAE)
            cd_MAE=count_cd(MAE, len(MAE[0]))
            nem_MAE = NemFTest(MAE_DF)
            print('CD_MAE: ', cd_MAE)
            print('MAE: ', NemFTest(MAE_DF))
            average_by_data_MAE = make_rank(MAE_DF)
            print('Average rank by data: ', average_by_data_MAE)
            average_by_PH_MAE = make_rank(nem_MAE)
            print('Average rank by Nemenyi test: ', average_by_PH_MAE)
            graph_cd(average_by_data_MAE, cd_MAE, 'MAE_by_data')
            graph_cd(average_by_PH_MAE, cd_MAE, 'MAE_by_test')

            graph_names.append('MAE_by_data')
            graph_names.append('MAE_by_test')

    return graph_names

    ###########################################################


def main():
    x = statistic_tests('C:/Users/eryk6/Downloads/test.csv', True, True, False, True, False, True, True, True, True, True)
    print()
    print(x)


if __name__ == '__main__': main()
