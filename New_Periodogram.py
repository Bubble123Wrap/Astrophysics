import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt  

#File Location
path = {'NGC1275':{3 : r"C:\Users\abhay\OneDrive\Desktop\Everything\Academic\Intern_Sem7\Astro_Gaussian_analysis\Data\NGC 1275\4FGL_J0319.8+4130_daily_3_6_2023.csv"
                 , 7 : r"C:\Users\abhay\OneDrive\Desktop\Everything\Academic\Intern_Sem7\Astro_Gaussian_analysis\Data\NGC 1275\4FGL_J0319.8+4130_weekly_31_5_2023.csv"
                 , 30 : r"C:\Users\abhay\OneDrive\Desktop\Everything\Academic\Intern_Sem7\Astro_Gaussian_analysis\Data\NGC 1275\4FGL_J0319.8+4130_monthly_3_6_2023.csv"}
       ,'PKS1510-089':{3 : r"C:\Users\abhay\OneDrive\Desktop\Everything\Academic\Intern_Sem7\Astro_Gaussian_analysis\Data\PKS 1510-089\4FGL_J1512.8-0906_daily_3_6_2023.csv"
                 , 7 : r"C:\Users\abhay\OneDrive\Desktop\Everything\Academic\Intern_Sem7\Astro_Gaussian_analysis\Data\PKS 1510-089\4FGL_J1512.8-0906_weekly_31_5_2023.csv"
                 , 30 : r"C:\Users\abhay\OneDrive\Desktop\Everything\Academic\Intern_Sem7\Astro_Gaussian_analysis\Data\PKS 1510-089\4FGL_J1512.8-0906_monthly_3_6_2023.csv"}}

#Read CSV
def csv_reader(path):
    df = {}
    for i in path:
        cache = {}
        for j in path[i]:
            cache.update({j : pd.read_csv(path[i][j])})
        df.update({i : cache})
    return df

df = csv_reader(path)

#Removing Unwanted Data Points
def DataExt(df):
    Data = {}
    for i in df:
        cache = {}
        for j in df[i]:
            info = []
            for n,m in zip(df[i][j]['Julian Date'],df[i][j]['Photon Flux [0.1-100 GeV](photons cm-2 s-1)']):
                try :
                    info.append([n,float(m)])
                except :
                    pass
            cache.update({j : info})
        Data.update({i : cache})
    return Data

Data = DataExt(df)

#Calculating frequency and power using astropy
def FPExt(Data):
    FP = {}
    for i in Data:
        cache = {}
        for j in Data[i]:
            frequency, power = LombScargle(np.array(Data[i][j])[:,0], np.array(Data[i][j])[:,1]).autopower()
            cache[j] = {'Frequency' : frequency , 'Power' : power}
        FP[i] = cache
    return FP

FP = FPExt(Data)

for i in Data:
    for j in Data[i]:
        fig, (ax1,ax2) = plt.subplots(2)
        ax1.plot(np.array(Data[i][j])[:,0],np.array(Data[i][j])[:,1])
        ax2.plot(FP[i][j]['Frequency'],FP[i][j]['Power'])
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.title(f'{i}: Sampling Period {j}')
        plt.show()
        plt.show()



