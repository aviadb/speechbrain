from numpy import NaN
import torch

class SpectrumScaler:
    pass

class MelScaler():
    def __init__(self):
        pass

    def hz2mel(self, f):
        pass


class BarkScaler():
    def __init__(self):
        pass

    def hz2bark(self, f, method=2, fix=False):
        if method == 1:
            (bark, c_bands) = self._m1_hz2bark(f)
        elif method == 2:
            (bark, c_bands) = self._m2_hz2bark(f, fix)
        elif method == 3:
            (bark, c_bands) = self._m3_hz2bark(f)
        return (bark, c_bands)

    def _espnet_hz2bark(self, f):
        pass

    def bark2hz(self, bark, method=1, fix=False):
        if method == 1:
            hz = self._m1_bark2hz(bark, fix)
        elif method == 2:
            hz = self._m2_bark2hz(bark)
        elif method == 3:
            hz = self._m3_bark2hz(bark)

        return hz

    def _m1_hz2bark(self, f): # 'z'
        # [2] E. Zwicker, Subdivision of the audible frequency range into
        # critical bands, J Accoust Soc Am 33, 1961, p248.
        bark = torch.pow(torch.div(f, 7500), 2)
        bark = 13 * torch.arctan(0.76e-3*f) + 3.5 * torch.arctan(bark)
        
        c_bands = torch.pow(1 + torch.mul(1.4e-6, torch.pow(f, 2)), 0.69)
        c_bands = 25 + torch.mul(c_bands, 75)
        return bark, c_bands

    def _m2_hz2bark(self, f, fix=False): # 'default', 'lhLH'
        # [1] H. Traunmuller, Analytical Expressions for the
        # Tonotopic Sensory Scale�, J. Acoust. Soc. Am. 88,
        # 1990, pp. 97-100.
        bark = torch.div(26.81*f, (1960+f)) - 0.53
        c_bands =  1 / ((26.81 * 1960) * torch.sqrt(1960+f)) 

        if fix:
            bark[bark < 2] = 0.3 + 0.85 * bark[bark < 2]
            bark[bark > 20.1] = bark[bark > 20.1] + 0.22*(bark[bark > 20.1]-20.1)

            c_bands[bark < 2] = 0.85 * c_bands[bark < 2]
            c_bands[bark > 20.1] = 1.22 * c_bands[bark > 20.1]
        else:
            m1 = (bark < 3)
            P = 0.53 / (3.53)**2
            bark[m1] = bark[m1] + (P * torch.pow((3-bark[m1]), 2))

            m2 = (bark > 21.4)
            m1 = torch.logical_and((bark > 19.4), ~m2)
            bark[m1] = bark[m1] + 0.0625 * torch.pow((bark[m1]-19.4), 2)
            bark[m2] = (1 + 0.25) * bark[m2] - 0.25*20.4

            c_bands[m1] = torch.mul(c_bands[m1], 1+2*0.0625*(bark[m1]-19.4))
            c_bands[m2] = torch.mul(c_bands[m2], 1.25)

        return bark, c_bands

    def _m3_hz2bark(self, f): # 's'
        # [3] M. R. Schroeder, B. S. Atal, and J. L. Hall. Optimizing digital
        # speech coders by exploiting masking properties of the human ear.
        # J. Acoust Soc Amer, 66 (6): 1647ï¿½1652, 1979. doi: 10.1121/1.383662.
        bark = 7 * torch.log(f/650 + torch.sqrt(1 + torch.pow((f / 650), 2)))
        # bark = 6 * torch.asinh(torch.div(f, 600))

        c_bands = torch.cosh(torch.div(bark, 7))
        c_bands = torch.mul(c_bands, 650/7)
        return bark, c_bands

    def _m1_bark2hz(self, bark, fix=False): # 'default'
        if fix:
            bark[bark < 2] = (bark[bark < 2] - 0.3) / 0.85
            bark[bark > 20.1] = (bark[bark > 20.1] + 4.422) / 1.22
        else:
            m1 = (bark < 3)
            P = 0.53 / (3.53)**2
            V = 3 - (0.5/P)
            W = V**2 - 9
            bark[m1] = V + torch.sqrt(W + (bark[m1]/P))

            m2 = (bark > 21.65)
            m1 = torch.logical_and((bark > 19.4), ~m2)
            bark[m2] = (bark[m2] + 0.25*20.4) / 1.25
            bark[m1] = 11.4 + torch.sqrt(-246.4 + (bark[m1]/0.0625))

        hz = ((26.81 * 1960) * 1/(26.81 - bark - 0.53)) - 1960

        return hz

    def _m2_bark2hz(self, bark, fix=False):
        hz = 1

        return hz

    def _m3_bark2hz(self, bark): # 's'
        # [3] M. R. Schroeder, B. S. Atal, and J. L. Hall. Optimizing digital
        # speech coders by exploiting masking properties of the human ear.
        # J. Acoust Soc Amer, 66 (6): 1647�1652, 1979. doi: 10.1121/1.383662.       
        hz = 650 * torch.asinh(torch.div(bark, 7))

        return hz

class ERBScaler():
    def __init__(self):
        pass

    def hz2erb(self, f, approx=False):
        if approx:
            erb = 21.4 * torch.log10(1 + torch.mul(0.00437, f))
        else:
            erb = torch.mul(11.17, torch.log(47.065 - torch.div(676170.42, f + 14678.5))) 

        return erb, NaN

    def erb2hz(self, erb, approx=False):
        if approx:
            hz = torch.div(torch.pow(10, torch.div(erb, 21.4)) - 1, 0.00437)
        else:
            hz = torch.div(676170.4, torch.maximum((47.06538 - torch.exp(torch.mul(0.0895, torch.abs(erb)))), torch.tensor(0))) - 14678.5

        return hz