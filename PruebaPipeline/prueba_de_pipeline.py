# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np 
from datetime import datetime
import ee
import pickle
service_account = 'pipelineaquosmic@persuasive-ego-353300.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'persuasive-ego-353300-8f6f4f635741.json')
ee.Initialize(credentials)

#Recopilación de datos

##Longitud, Latitud y Fecha

lonarr=[]
latarr=[]
datearr=[]
n=st.text_input("¿Cuántos puntos tienes?", )
for i in range(0,int(n)):
  lon=st.text_input("Longitud del punto "+str(i+1), )
  lonarr.append(float(lon))
  lat=st.text_input("Latitud del punto "+str(i+1), )
  latarr.append(float(lat))
  day=st.text_input("Dia del punto "+str(i+1), )
  month=st.text_input("Mes del punto "+str(i+1), )
  year=st.text_input("Año del punto "+str(i+1), )
  date=year+'-'+month+'-'+day
  print(date)
  date=datetime.strptime(date, '%Y-%m-%d')
  datearr.append(date)
lon_pd = pd.DataFrame(lonarr)
lon_pd.columns = ['Lon']
lat_pd = pd.DataFrame(latarr)
lat_pd.columns = ['Lat']
date_pd = pd.DataFrame(datearr)
date_pd.columns = ['Date']
Base=lon_pd.join(lat_pd)
Base=Base.join(date_pd)
Base

##Bandas

date_plus = []
data_1f = Base.reset_index()  # make sure indexes pair with number of rows


for index, row in data_1f.iterrows():
    date_row = row['Date']
    date_plus.append(date_row)


date_pd = pd.DataFrame(date_plus)
date_pd.columns = ['Date_ref']

data_1j = Base.join(date_pd)

data_1j['Date_plus'] = data_1j['Date_ref'] + pd.DateOffset(months=1)
data_1j['Date_less'] = data_1j['Date_ref'] + pd.DateOffset(months=-1)

data_1j.info()

B1 = []
B2 = []
B3 = []
B4 = []
B5 = []
B6 = []
B7 = []
B8 = []
B9 = []
B10 = []
B11 = []
ActDate = []

#https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1
landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1')
for index, row in data_1j.iterrows():
  try:
    landsat_1f = landsat.filterDate(row['Date_less'],row['Date_plus'])
    Image_a = ee.Image(landsat_1f.filterBounds(ee.Geometry.Point([row['Lon'], row['Lat']])).sort('CLOUD_COVER').first())
    p = ee.Geometry.Point(row['Lon'], row['Lat'])
    meanDictionary = Image_a.reduceRegion(reducer=ee.Reducer.mean(), geometry=p, scale=1)
    u = meanDictionary.getInfo()
    B1.append(u['B1'])
    B2.append(u['B2'])
    B3.append(u['B3'])
    B4.append(u['B4'])
    B5.append(u['B5'])
    B6.append(u['B6'])
    B7.append(u['B7'])
    B8.append(u['B8'])
    B9.append(u['B9'])
    B10.append(u['B10'])
    B11.append(u['B11'])
    date1 = ee.Date(landsat_1f.filterBounds(ee.Geometry.Point([row['Lon'], row['Lat']])).sort('CLOUD_COVER').first().get('system:time_start'))
    date2 = date1.format('Y-MM-dd').getInfo()
    ActDate.append(date2)
    print(index)
  except:
    B1.append('')
    B2.append('')
    B3.append('')
    B4.append('')
    B5.append('')
    B6.append('')
    B7.append('')
    B8.append('')
    B9.append('')
    B10.append('')
    B11.append('')
    ActDate.append('')

b1_pd = pd.DataFrame(B1)
b1_pd.columns = ['B1']
b2_pd = pd.DataFrame(B2)
b2_pd.columns = ['B2']
b3_pd = pd.DataFrame(B3)
b3_pd.columns = ['B3']
b4_pd = pd.DataFrame(B4)
b4_pd.columns = ['B4']
b5_pd = pd.DataFrame(B5)
b5_pd.columns = ['B5']
b6_pd = pd.DataFrame(B6)
b6_pd.columns = ['B6']
b7_pd = pd.DataFrame(B7)
b7_pd.columns = ['B7']
b8_pd = pd.DataFrame(B8)
b8_pd.columns = ['B8']
b9_pd = pd.DataFrame(B9)
b9_pd.columns = ['B9']
b10_pd = pd.DataFrame(B10)
b10_pd.columns = ['B10']
b11_pd = pd.DataFrame(B11)
b11_pd.columns = ['B11']
actdate_pd = pd.DataFrame(ActDate)
actdate_pd.columns = ['Actual_Date']

data_2j = data_1j.join(actdate_pd)
data_2j = data_2j.join(b1_pd)
data_2j = data_2j.join(b2_pd)
data_2j = data_2j.join(b3_pd)
data_2j = data_2j.join(b4_pd)
data_2j = data_2j.join(b5_pd)
data_2j = data_2j.join(b6_pd)
data_2j = data_2j.join(b7_pd)
data_2j = data_2j.join(b8_pd)
data_2j = data_2j.join(b9_pd)
data_2j = data_2j.join(b10_pd)
data_2j = data_2j.join(b11_pd)

BaseLimpiaBandas=data_2j
BaseLimpiaBandas

col=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
for i in col:
  BaseLimpiaBandas[i]=BaseLimpiaBandas[i].apply(lambda x: x/65455)
BaseLimpiaBandas

#Predicciones
##Modelos

lrOUSA=pickle.load(open('LinearOxigenoUSA.sav','rb'))
lrNUSA=pickle.load(open('LinearNitrogenoUSA.sav','rb'))
lrFUSA=pickle.load(open('LinearFosforoUSA.sav','rb'))
lrCOUSA=pickle.load(open('LinearConductividadUSA.sav','rb'))
lrCLUSA=pickle.load(open('LinearClorofilaUSA.sav','rb'))
lrOMex=pickle.load(open('LinearOxigenoMéxicoLent.sav','rb'))
lrNMex=pickle.load(open('LinearNitrogenoMéxicoLent.sav','rb'))
lrCOMex=pickle.load(open('LinearConductividadMéxicoLent.sav','rb'))
lrTMex=pickle.load(open('LinearTurbiedadMéxicoLent.sav','rb'))
lrFMex=pickle.load(open('LinearFosforoMéxicoLent.sav','rb'))

#Tabla

OxiUSAArr=[]
OxiMexArr=[]
NitUSAArr=[]
NitMexArr=[]
CondUSAArr=[]
CondMexArr=[]
FosfUSAArr=[]
FosfMexArr=[]
ClorofUSAArr=[]
TurbMexArr=[]
for i in range(0,int(n)):
  OxiUSAArr.append(lrOUSA.predict(np.array(BaseLimpiaBandas[['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']].iloc[i]).reshape(1,-1)))
  OxiMexArr.append(lrOMex.predict(np.array(BaseLimpiaBandas[['B2','B3','B4','B5','B6','B8']].iloc[i]).reshape(1,-1)))
  NitUSAArr.append(lrNUSA.predict(np.array(BaseLimpiaBandas[['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']].iloc[i]).reshape(1,-1)))
  NitMexArr.append(lrNMex.predict(np.array(BaseLimpiaBandas[['B2','B3','B4','B5','B6','B8']].iloc[i]).reshape(1,-1)))
  FosfUSAArr.append(lrFUSA.predict(np.array(BaseLimpiaBandas[['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']].iloc[i]).reshape(1,-1)))
  FosfMexArr.append(lrFMex.predict(np.array(BaseLimpiaBandas[['B1','B2','B3','B4','B5','B6','B8']].iloc[i]).reshape(1,-1)))
  CondUSAArr.append(lrCOUSA.predict(np.array(BaseLimpiaBandas[['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']].iloc[i]).reshape(1,-1)))
  CondMexArr.append(lrCOMex.predict(np.array(BaseLimpiaBandas[['B1','B2','B3','B4','B5','B6','B8']].iloc[i]).reshape(1,-1)))
  ClorofUSAArr.append(lrCLUSA.predict(np.array(BaseLimpiaBandas[['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']].iloc[i]).reshape(1,-1)))
  TurbMexArr.append(lrTMex.predict(np.array(BaseLimpiaBandas[['B1','B2','B3','B4','B5','B6','B8']].iloc[i]).reshape(1,-1)))


OxiUSA_pd = pd.DataFrame(OxiUSAArr)
OxiUSA_pd.columns = ['Oxigeno USA mg/L']
OxiMex_pd = pd.DataFrame(OxiMexArr)
OxiMex_pd.columns = ['Oxigeno Mex mg/L']
NitUSA_pd = pd.DataFrame(NitUSAArr)
NitUSA_pd.columns = ['Nitrogeno USA Ammonia Total mg/L']
NitMex_pd = pd.DataFrame(NitMexArr)
NitMex_pd.columns = ['Nitrogeno Mex Total mg/L']
FosfUSA_pd = pd.DataFrame(FosfUSAArr)
FosfUSA_pd.columns = ['Fosforo USA Total mg/L']
FosfMex_pd = pd.DataFrame(FosfMexArr)
FosfMex_pd.columns = ['Fosforo Mex Total mg/L']
CondUSA_pd = pd.DataFrame(CondUSAArr)
CondUSA_pd.columns = ['Conductividad USA por sólidos mg/L']
CondMex_pd = pd.DataFrame(CondMexArr)
CondMex_pd.columns = ['Conductividad Mex electrico microS/cm']
TurbMex_pd = pd.DataFrame(TurbMexArr)
TurbMex_pd.columns = ['Turbiedad Mex UNT']
ClorofUSA_pd = pd.DataFrame(ClorofUSAArr)
ClorofUSA_pd.columns = ['Clorofila USA ug/L']

Tabla=Base.join(OxiUSA_pd)
Tabla=Tabla.join(OxiMex_pd)
Tabla=Tabla.join(NitUSA_pd)
Tabla=Tabla.join(NitMex_pd)
Tabla=Tabla.join(FosfUSA_pd)
Tabla=Tabla.join(FosfMex_pd)
Tabla=Tabla.join(CondUSA_pd)
Tabla=Tabla.join(CondMex_pd)
Tabla=Tabla.join(ClorofUSA_pd)
Tabla=Tabla.join(TurbMex_pd)

Tabla