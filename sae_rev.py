import math as mt

import numpy as np
import pandas as pd
from scipy.stats import linregress

path = 'data/'
dict_immat = {}
for i in range(10):
    dict_immat['data_immat_' + str(i)] = pd.read_csv(
        path + 'data_immat_traitees_' + str(i) + '.csv', sep=';', low_memory=False)

data_immat = pd.concat(dict_immat, ignore_index=True)
data_communes = pd.read_csv(path + 'RGC_2013.csv', sep=';', low_memory=False)
data_bornes = pd.read_csv(
    path + 'data_bornes_traitees.csv', sep=';', low_memory=False)

data_communes['DEP'] = data_communes['DEP'].astype(str)
data_communes['COM'] = data_communes['COM'].astype(str)
data_communes['COG'] = data_communes['DEP'] + data_communes['COM'].str.zfill(3)
data_immat['code commune'] = data_immat['code commune'].astype(str)
data_bornes['code_insee_commune'] = data_bornes['code_insee_commune'].astype(
    str)
data_bornes['dep_b'] = data_bornes['dep_b'].astype(int).astype(str)
data_immat['dptmt'] = data_immat['dptmt'].astype(str)


def calcul_ratio(list_dep):
    data_dep = data_immat[data_immat["dptmt"].isin(list_dep)]
    data_dep = data_dep[data_dep["energie"] == "Electrique et hydrogene"]
    data_dep = data_dep.groupby("dptmt").sum(True)
    data_bornes_dep = data_bornes[data_bornes["dep_b"].isin(list_dep)]
    data_bornes_dep = data_bornes_dep.groupby("dep_b").sum(True)

    return data_dep["Nb_immat"] / data_bornes_dep["Nombre points de charge"]


def repart_ratio():
    data_popu = data_communes[data_communes["POPU"] > 0]
    data_immat_by_communes = data_immat.groupby("code commune").sum(True)

    data_merge = data_immat_by_communes.merge(data_popu, "inner", left_on="code commune", right_on="COG")

    data_merge["ratio"] = data_merge["Nb_immat"] / data_merge["POPU"] * 10 / 13
    data_merge = data_merge[data_merge["ratio"] <= 100]

    data_merge_admi = data_merge.groupby("ADMI").sum(True)

    data_merge_admi["ratio"] = data_merge_admi["Nb_immat"] / data_merge_admi["POPU"] * 10 / 13
    data_merge_admi = data_merge_admi[data_merge_admi["ratio"] <= 100]

    return data_merge["ratio"], data_merge_admi["ratio"]


def repart_energie(annee):
    data_energie = data_immat[data_immat["Annee"] == annee]
    data_energie = data_energie.groupby("energie")["Nb_immat"].sum()
    return data_energie


def get_tx_elec():
    data = data_immat[["energie", 'Annee', "Nb_immat"]]
    data = data.groupby(['Annee', "energie"])["Nb_immat"].sum().reset_index()
    data_eletrique = data[data["energie"] == "Electrique et hydrogene"]
    data_eletrique = data_eletrique.groupby("Annee").sum(True)

    data_toutes_energie = data.groupby("Annee").sum()

    return data_eletrique["Nb_immat"] / data_toutes_energie["Nb_immat"]


def mod_log_evol_tx_elec(tx_sat, x_mod_log, serie_tx_elec):
    serie = []
    for i in range(0, len(serie_tx_elec)):
        serie.append(mt.log(tx_sat / serie_tx_elec.values[i] - 1))
    res = linregress(serie_tx_elec.index, serie)
    rep = []
    for j in range(0, len(x_mod_log)):
        rep.append(res.slope * x_mod_log[j] + res.intercept)
    return rep, res.slope


# QUELQUES TESTS INDICATIFS
test_calcul_ratio = calcul_ratio(['44', '72', '85', '53'])
print('test calcul_ratio : ', np.isclose(test_calcul_ratio.sum(), 20.83005552505825)
      and np.isclose(test_calcul_ratio.loc['72'], 5.063337393422655))

test_repart_energie = repart_energie(2017)
print('test_repart_energie : ', test_repart_energie.loc['Diesel - thermique'] == 1013098)

serie_ratio_com, serie_ratio_admi = repart_ratio()


print('test_repart_ratio1 : ', np.isclose(
    serie_ratio_com.mean(), 19.887652614523343))
print('test_repart_ratio2 : ', np.isclose(
    serie_ratio_admi.loc[3], 34.55047603236587))

serie_tx_elec = get_tx_elec()

print('test_get_tx_elec : ', np.isclose(serie_tx_elec.mean(
), 0.028857941346249236) and np.isclose(serie_tx_elec.loc[2020], 0.0665604078072506))
