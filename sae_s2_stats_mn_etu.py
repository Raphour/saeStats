import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipf, linregress
import math as mt

path = 'data/'
dict_immat = {}
for i in range(10):
    dict_immat['data_immat_'+str(i)] = pd.read_csv(
        path+'data_immat_traitees_'+str(i)+'.csv', sep=';', low_memory=False)

data_immat = pd.concat(dict_immat, ignore_index=True)
data_communes = pd.read_csv(path+'RGC_2013.csv', sep=';', low_memory=False)
data_bornes = pd.read_csv(
    path+'data_bornes_traitees.csv', sep=';', low_memory=False)

data_communes['DEP'] = data_communes['DEP'].astype(str)
data_communes['COM'] = data_communes['COM'].astype(str)
data_communes['COG'] = data_communes['DEP']+data_communes['COM'].str.zfill(3)
data_immat['code commune'] = data_immat['code commune'].astype(str)
data_bornes['code_insee_commune'] = data_bornes['code_insee_commune'].astype(
    str)
data_bornes['dep_b'] = data_bornes['dep_b'].astype(int).astype(str)
data_immat['dptmt'] = data_immat['dptmt'].astype(str)


def calcul_ratio(list_dep):
    data_voiture_electrique = data_immat[data_immat['energie']
                                         == 'Electrique et hydrogene']
    data_voiture_electrique = data_voiture_electrique[data_voiture_electrique['dptmt'].isin(
        list_dep)]
    data_voiture_electrique = data_voiture_electrique.groupby(
        ['dptmt']).sum(numeric_only=True)
    data_voiture_electrique = data_voiture_electrique['Nb_immat']
    data_bornes_dep = data_bornes[data_bornes['dep_b'].isin(list_dep)]
    data_bornes_dep = data_bornes_dep.groupby(['dep_b']).sum(numeric_only=True)

    data_bornes_dep = data_bornes_dep['Nombre points de charge']

    ratio = data_voiture_electrique/data_bornes_dep

    return ratio


calcul_ratio(['44', '49', '72', '85', '53'])


def repart_energie(annee):
    # Sélectionner les données pour l'année donnée en paramètre
    immatriculations_annee = data_immat[data_immat['Annee'] == annee]
    # Regrouper les données par type de carburant et calculer le nombre total d'immatriculations pour chaque type de carburant
    repartition_energie = immatriculations_annee.groupby(['energie']).sum(numeric_only=True)
    repartition_energie  = repartition_energie['Nb_immat']
    return repartition_energie





def repart_ratio(data_communes, data_immat):
    # Somme des immatriculations par commune
    immat_by_commune = data_immat.groupby('code commune')['Nb_immat'].sum()
    # Ajouter la colonne 'code commune' basée sur la concaténation des colonnes 'DEP' et 'COM' dans data_communes
    data_communes['code commune'] = data_communes['DEP'] + data_communes['COM'].str.zfill(3)
    # Fusionner avec data_communes pour obtenir la population et le statut administratif
    data_merged = data_communes.merge(immat_by_commune, on='code commune')
    # Calculer le ratio du nombre d'immatriculations par centaine d'habitants sur 10 ans pour chaque commune
    data_merged['ratio'] = (data_merged['Nb_immat'] / (data_merged['POPU'])) * (10 / 13) # 10 ans sur une période de 12 ans
    # Exclure les communes ayant une population égale à zéro
    data_merged = data_merged[data_merged['POPU'] != 0]
    # Restriction aux communes ayant un ratio inférieur ou égal à 100
    data_merged = data_merged[data_merged['ratio'] <= 100]
    # Calculer le ratio sur l'ensemble des communes regroupées par statut administratif
    total_immat_by_statut = data_merged.groupby('ADMI')['Nb_immat'].sum()
    total_popu_by_statut = data_merged.groupby('ADMI')['POPU'].sum()
    ratios_by_statut = (total_immat_by_statut / (total_popu_by_statut)) * (10 / 13) # 10 ans sur une période de 12 ans
    # Renvoyer les résultats sous forme de séries
    ratios_all_communes = data_merged['ratio']
    ratios_by_statut = ratios_by_statut.rename('ratio').rename_axis('ADMI')

    return ratios_all_communes, ratios_by_statut






def get_tx_elec():
    return  # serie_tx


def mod_log_evol_tx_elec(tx_sat, x_mod_log, serie_tx_elec):
    return  # vecteur_mod_log


def mod_log_rec(u0, n, k, a):
    return  # liste_valeurs


def BaseLagrange(x, listX, i):
    return  # une valeur


def InterLagrange(x, listX, listY):
    return  # une valeur


def dicho(a, b, f, e):
    return  # liste_valeurs


def fausse_pos(a, b, f, e):
    return  # liste_valeurs


# QUELQUES TESTS INDICATIFS
# test_calcul_ratio = calcul_ratio(['44', '72', '85', '53'])
# print('test calcul_ratio : ', np.isclose(test_calcul_ratio.sum(), 20.83005552505825)
#       and np.isclose(test_calcul_ratio.loc['72'], 5.063337393422655))

# test_repart_energie = repart_energie(2017)
# print('test_repart_energie : ',
#       test_repart_energie.loc['Diesel - thermique'] == 1013098)

serie_ratio_com,serie_ratio_admi=repart_ratio(data_communes,data_immat)
print(serie_ratio_admi)
print('====')
print(serie_ratio_com)
print("MOYENNE",serie_ratio_com.mean())
print('test_repart_ratio1 : ',np.isclose(serie_ratio_com.mean(),19.887652614523343))
print('test_repart_ratio2 : ',np.isclose(serie_ratio_admi.loc[3],34.55047603236587))

# serie_tx_elec=get_tx_elec()
# print('test_get_tx_elec : ',np.isclose(serie_tx_elec.mean(),0.028857941346249236) and np.isclose(serie_tx_elec.loc[2020],0.0665604078072506))

# serie_tx_elec_entree=pd.Series([0.010786,0.011784,0.014267,0.019319,0.066560],index=np.arange(2016,2021))
# serie_tx_mod_log,a=mod_log_evol_tx_elec(0.8,np.linspace(2010,2050,100),serie_tx_elec_entree)
# print('test mod_log_evol_tx_elec : ',np.isclose(serie_tx_mod_log.mean(),0.4658567447364723))

# tx2010=8.224631848765294e-05
# n2030=20
# tx_sat=0.8
# a=-0.478251476283651
# y=mod_log_rec(tx2010,n2030,tx_sat,a)
# list_renv=[8.224631848765294e-05, 0.00013267560116676642, 0.00021401729659258336, 0.0003452070109277549, 0.0005567582681753597, 0.0008978077274020229, 0.0014473931927825384, 0.00233242083301888, 0.0037560648821641104, 0.0060420791637187075, 0.009702452752420297, 0.015536936094658503, 0.024769967993597573, 0.03921556294204935, 0.061418242859102315, 0.09462787378608183, 0.14233386555121186, 0.20703147763038865, 0.28824798529356527, 0.3808611222245485, 0.4755776512869108]
# print('test mod_log_rec : ',list(np.isclose(np.array(y),np.array(list_renv))).count(True)==len(y))

# print('test BaseLagrange : ',np.isclose(BaseLagrange(0.2,np.linspace(-1,1,10),6),0.37638881280000036))
# print('test2 BaseLagrange : ',np.isclose(BaseLagrange(0.2,np.linspace(-1,1,11),6),1))
# print('test3 BaseLagrange : ',np.isclose(BaseLagrange(0.2,np.linspace(-1,1,11),4),0))

# print('test1 InterLagrange : ',np.isclose(InterLagrange(np.pi/3,np.linspace(0,np.pi,7),np.cos(np.linspace(0,np.pi,7))),0.5))
# print('test2 InterLagrange : ',np.isclose(InterLagrange(np.pi/3,np.linspace(0,np.pi,5),np.cos(np.linspace(0,np.pi,5))),0.4969732592091243))

# list_res_dicho1=[1.5, 1.25, 1.375, 1.4375, 1.40625, 1.421875, 1.4140625, 1.41796875, 1.416015625, 1.4150390625, 1.41455078125, 1.414306640625, 1.4141845703125, 1.41424560546875, 1.414215087890625]
# print('test dicho : ',list(np.isclose(np.array(dicho(1,2,lambda x:x**2-2,10**(-5))),np.array(list_res_dicho1))).count(True)==len(list_res_dicho1))

# list_res_fausse_pos1=[1.3333333333333333, 1.4, 1.411764705882353, 1.4137931034482758, 1.414141414141414, 1.4142011834319526, 1.41421143847487]
# print('test fausse_pos : ',list(np.isclose(np.array(fausse_pos(1,2,lambda x:x**2-2,10**(-5))),np.array(list_res_fausse_pos1))).count(True)==len(list_res_fausse_pos1))


def test_remarque(annee_debut, annee_fin, tx_sat, serie_tx_elec, e, tx_recherche):
    N = annee_fin-annee_debut+1
    absc_interp_cheb = np.array((0.5*(annee_debut+annee_fin))*np.ones(N))+(0.5*(
        annee_fin-annee_debut))*np.cos(np.array([mt.pi*(2*k-1)/(2*N) for k in range(1, N+1)]))
    listY, a = mod_log_evol_tx_elec(tx_sat, absc_interp_cheb, serie_tx_elec)
    annee_debut_serie = serie_tx_elec.index.values.min()
    res_mod_log_rec = mod_log_rec(
        serie_tx_elec.loc[annee_debut_serie], annee_fin-annee_debut, tx_sat, a)
    print('test_remarque mod_log_rec : ', annee_debut_serie +
          [el < tx_recherche for el in res_mod_log_rec].count(True))
    print('test_remarque_dicho : ', dicho(annee_debut, annee_fin,
          lambda x: InterLagrange(x, absc_interp_cheb, listY)-tx_recherche, e))
    print('test_remarque_fausse_pos : ', fausse_pos(annee_debut, annee_fin,
          lambda x: InterLagrange(x, absc_interp_cheb, listY)-tx_recherche, e))

# test_remarque(2010,2050,0.8,get_tx_elec().loc[2011:2022],10**(-3),0.5) #doit afficher :
# test_remarque mod_log_rec :  2029
# test_remarque_dicho :  [2030.0, 2020.0, 2025.0, 2027.5, 2028.75, 2028.125]
# test_remarque_fausse_pos :  [2034.9828526294257, 2026.228493362021, 2029.3131503768009, 2028.2055530020161, 2028.1212100178818]
