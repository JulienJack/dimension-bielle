import numpy as np
from numpy import sqrt,cos,sin,pi
from scipy.integrate import solve_ivp

tau = 10 #[de 8 à 13 pour les modernes] = Volume au point mort bas PMB (vol max) / volume au point mort haut PMH (vol min)
# pour l essence le taux vaire de 1.0 a 1.2 MPa et pour le diesel de 3.0 a 3.5 MPa
D = @valeur alesage@ #[m] = diametre du cylindre dans lequel le piston rentre
C = @valeur course@ #[m] = la course est le déplacement du piston donc max de 2R
Vc = pi*((D)**2)*D/4 #[-] = la cylindrée, soit le volume max - volume min

L = @valeur longueur bielle@  # [m] = la tige entre le piston et le vilebrequin ( relié au maneton du vilebrequin )
mpiston = @valeur masse piston@ #[kg] =
mbielle = @valeur masse bielle@ #[kg] = connecting rod
Q = 2 800 000 #[J/kg_inlet gas]

#Deux directions de flambages ?

R=C/2 @distance entre le maneton et le tourillon@ #en [m]
b=L/R

#theta en degre ss forme d un vecteur de - 180 a 180
#Mais on va simuler un cycle complet ? donc 2* 360 ?
theta= np.arange(-180*2,180*2,5)
V_output=np.zeros(len(theta))
def V_cyl(theta):
    for index,j in enumerate(theta):
        v=(Vc/2)*(1-cos(theta)+b-sqrt(b**2-sin(theta)**2))+(1*Vc)/(tau-1)
        V_output[index]=v
    return V_output

def derivee_Vcyl(theta):
    return (Vc/2)*(sin(theta)+(sin(theta)*cos(theta))/(sqrt(b**2-sin(theta)**2)))

def apport_chaleur(theta): # calculer l'apport de chaleur sur la durée de temps de combustion voir son schéma 
    return Q*(1-cos(pi*(theta-thetaC)/deltaThetaC))/2 # en [J]

def derivee_chaleur(theta):
    return Q*pi*sin(pi*(theta-thetaC)/deltaThetaC)/(2*deltaThetaC)

def derivee_pression(Theta,p):
    return ( -gamma*p*derivee_Vcyl(Theta) + (gamma-1)*derivee_chaleur(Theta) )/V_cycl(Theta)

def force_bielle():
    return F_pied_output et F_tete_output # en [N]

def pression_cylindre():
    return p_output # en [Pa]

def epaisseur_critique(): #? voir schema de l enonce
    return t  # en [m]


def myfunc(rpm, s, theta, thetaC, deltaThetaC):
    return (V_output, Q_output, F_pied_output, F_tete_output, p_output, t)  #v
