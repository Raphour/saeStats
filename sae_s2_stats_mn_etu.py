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


# def calcul_ratio(list_dep):
#     data_voit_elec = data_immat[data_immat["energie"] == 'Electrique et hydrogene']
#     data_voit_elec = data_voit_elec[data_voit_elec["dptmt"].isin(list_dep)]
#     data_voit_elec.groupby("dptmt").sum(numeric_only=True)
#     data_voit_elec = data_voit_elec["Nb_immat"]
#     data_bornes_dep = data_bornes[data_bornes["dep_b"].isin(list_dep)]
#     data_bornes_dep.groupby("dep_b").sum(numeric_only=True)
#     data_bornes_dep = data_bornes_dep["Nombre de point de charge"]
#
#     data_return = data_voit_elec / data_bornes_dep
#
#     return data_return


def calcul_ratio(list_dep):
    data_immat_dep = data_immat[data_immat["dptmt"].isin(list_dep)]
    data_immat_dep = data_immat_dep[data_immat_dep["energie"] == 'Electrique et hydrogene']
    data_immat_dep = data_immat_dep.groupby("dptmt").sum(numeric_only=True)

    data_borne = data_bornes[data_bornes["dep_b"].isin(list_dep)]
    data_borne = data_borne.groupby("dep_b").sum(True)

    data_return = data_immat_dep["Nb_immat"] / data_borne["Nombre points de charge"]
    print(data_return)
    return data_return


#
# def repart_energie(annee):
#     # Sélectionner les données pour l'année donnée en paramètre
#     immatriculations_annee = data_immat[data_immat['Annee'] == annee]
#     # Regrouper les données par type de carburant et calculer le nombre total d'immatriculations pour chaque type de
#     # carburant
#     repartition_energie = immatriculations_annee.groupby(['energie']).sum(numeric_only=True)
#     repartition_energie = repartition_energie['Nb_immat']
#     return repartition_energie
#

def repart_energie(annee):
    data_immat_annee = data_immat[data_immat["Annee"] == annee]
    data_immat_annee = data_immat_annee.groupby("energie").sum(True)

    return data_immat_annee["Nb_immat"]


# def repart_ratio(data_communes, data_immat):
#    # Somme des immatriculations par commune
#    immat_by_commune = data_immat.groupby('code commune')['Nb_immat'].sum()
#    # Ajouter la colonne 'code commune' basée sur la concaténation des colonnes 'DEP' et 'COM' dans data_communes
#    data_communes['code commune'] = data_communes['DEP'] + \
#        data_communes['COM'].str.zfill(3)
#    # Fusionner avec data_communes pour obtenir la population et le statut administratif
#    data_merged = data_communes.merge(immat_by_commune, on='code commune')
#
#    data_merged['ratio'] = (data_merged['Nb_immat'] /(data_merged['POPU'])) * (10 / 13)
#    data_merged = data_merged[data_merged['POPU'] != 0]
#    data_merged = data_merged[data_merged['ratio'] <= 100]
#    # Calculer le ratio sur l'ensemble des communes regroupées par statut administratif
#    total_immat_by_statut = data_merged.groupby('ADMI')['Nb_immat'].sum()
#    total_popu_by_statut = data_merged.groupby('ADMI')['POPU'].sum()
#    ratios_by_statut = (total_immat_by_statut /
#                        total_popu_by_statut) * (10 / 13)
#    # Renvoyer les résultats sous forme de séries
#    ratios_all_communes = data_merged['ratio']
#    ratios_by_statut = ratios_by_statut.rename('ratio').rename_axis('ADMI')

#
#    return ratios_all_communes, ratios_by_statut

def repart_ratio():
    data_popu = data_communes[data_communes["POPU"] > 0]

    data_immat_by_commune = data_immat.groupby("code commune")["Nb_immat"].sum()

    data_merge = data_popu.merge(data_immat_by_commune, "inner", left_on="COG", right_on="code commune")

    data_merge["ratio"] = data_merge["Nb_immat"] / data_merge["POPU"] * 10 / 13

    data_merge = data_merge[data_merge["ratio"] <= 100]

    data_ret = data_merge["ratio"]

    data_admi = data_merge.groupby("ADMI").sum(True)

    data_admi["ratio"] = data_admi["Nb_immat"] / data_admi["POPU"] * 10 / 13

    data_admi = data_admi[data_admi["ratio"] <= 100]

    return data_ret, data_admi["ratio"]


# def get_tx_elec():
#
#    df = data_immat[['Annee', 'energie', 'Nb_immat']]
#    counts = df.groupby(['Annee', 'energie'])['Nb_immat'].sum().reset_index()
#    total_counts = counts.groupby('Annee')['Nb_immat'].sum()
#    elec_hydro_counts = counts[counts['energie'].isin(['Electrique et hydrogene'])]
#    tx_elec = elec_hydro_counts.groupby('Annee')['Nb_immat'].sum() / total_counts
#
#    return tx_elec


def get_tx_elec():
    data = data_immat[["energie", 'Annee', "Nb_immat"]]
    data = data.groupby(['Annee', "energie"])["Nb_immat"].sum().reset_index()
    taux_energie_annee = data.groupby("Annee")["Nb_immat"].sum(True)
    taux_elec_annee = data[data["energie"] == "Electrique et hydrogene"]
    taux_elec_annee=taux_elec_annee.groupby("Annee")["Nb_immat"].sum(True)

    return taux_elec_annee / taux_energie_annee


#
# def mod_log_evol_tx_elec(tx_sat, x_mod_log, serie_tx_elec):
#     # On créé une nouvelle serie
#     new_serie = []
#     # BOUCLE 1
#     for i in range(0, len(serie_tx_elec)):
#         # Pour tout les taux de serie taux elec
#         # On fais le log qu'on ajoute a la liste
#         new_serie.append(mt.log(tx_sat / serie_tx_elec.values[i] - 1))
#     rep = []
#     # LINEREGRESS
#     result = linregress(serie_tx_elec.index, new_serie)
#
#     # BOUCLE 2
#     for i in range(0, len(x_mod_log)):
#         rep.append(tx_sat / (1 + mt.exp(result.slope * x_mod_log[i] + result.intercept)))
#     rep = np.array(rep)
#     return rep, result.slope

def mod_log_evol_tx_elec(tx_sat, x_mod_log, serie_tx_elec):
    serie = []
    for i in range(0, len(serie_tx_elec)):
        serie.append(mt.log(tx_sat / serie_tx_elec.values[i] - 1))
    res = linregress(serie_tx_elec.index, serie)
    rep = []
    for j in range(0, len(x_mod_log)):
        rep.append(tx_sat / (1 + mt.exp(res.slope * x_mod_log[j] + res.intercept)))
    rep = np.array(rep)

    return rep, res.slope


# QUELQUES TESTS INDICATIFS
# test_calcul_ratio = calcul_ratio(['44', '72', '85', '53'])
# print('test calcul_ratio : ', np.isclose(test_calcul_ratio.sum(), 20.83005552505825)
#       and np.isclose(test_calcul_ratio.loc['72'], 5.063337393422655))
#
# test_repart_energie = repart_energie(2017)
# print('test_repart_energie : ', test_repart_energie.loc['Diesel - thermique'] == 1013098)
#
# serie_ratio_com, serie_ratio_admi = repart_ratio()
# print(serie_ratio_admi)
# print('====')
# print(serie_ratio_com)
# print("MOYENNE", serie_ratio_com.mean())
# print('test_repart_ratio1 : ', np.isclose(
#     serie_ratio_com.mean(), 19.887652614523343))
# print('test_repart_ratio2 : ', np.isclose(
#     serie_ratio_admi.loc[3], 34.55047603236587))

serie_tx_elec = get_tx_elec()

print('test_get_tx_elec : ', np.isclose(serie_tx_elec.mean(
), 0.028857941346249236) and np.isclose(serie_tx_elec.loc[2020], 0.0665604078072506))

# serie_tx_elec_entree = pd.Series([0.010786, 0.011784, 0.014267, 0.019319, 0.066560], index=np.arange(2016, 2021))
# serie_tx_mod_log, a = mod_log_evol_tx_elec(0.8, np.linspace(2010, 2050, 100), serie_tx_elec_entree)
# print('test mod_log_evol_tx_elec : ', np.isclose(serie_tx_mod_log.mean(), 0.4658567447364723))

# tx2010=8.224631848765294e-05 n2030=20 tx_sat=0.8 a=-0.478251476283651 y=mod_log_rec(tx2010,n2030,tx_sat,
# a) list_renv=[8.224631848765294e-05, 0.00013267560116676642, 0.00021401729659258336, 0.0003452070109277549,
# 0.0005567582681753597, 0.0008978077274020229, 0.0014473931927825384, 0.00233242083301888, 0.0037560648821641104,
# 0.0060420791637187075, 0.009702452752420297, 0.015536936094658503, 0.024769967993597573, 0.03921556294204935,
# 0.061418242859102315, 0.09462787378608183, 0.14233386555121186, 0.20703147763038865, 0.28824798529356527,
# 0.3808611222245485, 0.4755776512869108] print('test mod_log_rec : ',list(np.isclose(np.array(y),
# np.array(list_renv))).count(True)==len(y))
