import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt  
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
from scipy import signal
from peakdetect import peakdetect

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

l = {'NGC1275':{3:{1:0.0417},7:{1:0.0177},30:{1:0.004}},
     'PKS1510-089':{3:{1:0.041},7:{1:0.017},30:{1:0.004}}}
           
for i in Data:
    for j in Data[i]:
        tstep = j
        sampling_freq = 1/tstep
        x = np.arange(np.array(Data[i][j])[0,0],np.array(Data[i][j])[-1,0]+1,j)
        N = len(x)

        Nyquist_freq = sampling_freq/2
        f0 = Nyquist_freq/8
        label = []
        for k in range(0,4):
            y = np.sin(2 * np.pi * (k+1)*f0 * x)

            fstep = sampling_freq / N
            f = np.linspace(0,(N-1)*fstep,N)
            fourier = np.fft.fft(y)
            fourier_mag = np.abs(fourier) / N
            fplot = f[0:int(N/2)]
            fmag = 2 * fourier_mag[0:int(N/2)]

            plt.plot(fplot,fmag)
            peaks = signal.find_peaks(fmag,threshold=0.1)
            freq_peaks = fplot[peaks[0]] 
            print(freq_peaks)
            for m in freq_peaks:
                label.append(f'real:{int((k+1)*f0*1000)/1000},fourier:{int(m*1000)/1000}')
        label.insert(1,f'real:{int((1+1)*f0*1000)/1000},fourier:{l[i][j][1]}')     
        plt.title(f'{i}: Spacing {j} days')
        plt.legend(label,loc='upper right')
        plt.grid()
        plt.show()

    


       

