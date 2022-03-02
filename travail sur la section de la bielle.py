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
def volume(theta):
    for index,j in enumerate(theta):
        v=(Vc/2)*(1-cos(theta)+b-sqrt(b**2-sin(theta)**2))+(1*Vc)/(tau-1)
        V_output[index]=v
    return V_output

def derivee_volume(theta):
    return (Vc/2)*(sin(theta)+(sin(theta)*cos(theta))/(sqrt(b**2-sin(theta)**2)))

def apport_chaleur(theta): # calculer l'apport de chaleur sur la durée de temps de combustion voir son schéma
    if type(theta) != np.float64 and type(theta) != type(0.0) :
        theta[(thetaC > theta) or (theta > thetaC + deltaThetaC)] = 0.0
    elif (thetaC > theta) or (theta > thetaC + deltaThetaC) : return 0.0
    return Q*(1-cos(pi*(theta-thetaC)/deltaThetaC))/2 # en [J]

def derivee_chaleur(theta):
    if type(theta) != np.float64 and type(theta) != type(0.0) :
        theta[(thetaC > theta) or (theta > thetaC + deltaThetaC)] = 0.0
    elif (thetaC > theta) or (theta > thetaC + deltaThetaC) : return 0.0
    return Q*pi*sin(pi*(theta-thetaC)/deltaThetaC)/(2*deltaThetaC)

def derivee_pression(Theta,p):
    return ( -gamma*p*derivee_volume(Theta) + (gamma-1)*derivee_chaleur(Theta) )/volume(Theta)

def force_bielle():
    return (F_pied_output,F_tete_output) # en [N]

def pression_cylindre(theta):
    result = solve_ivp(derivee_pression,(-pi,pi),np.array([s*10**5]),t_eval = theta)
    #if(result.t != theta) : print("Attention aux angles évalués")
    return result.y.reshape(len(result.y[0])) # en [Pa]

def epaisseur_critique(): #? voir schema de l enonce
    return t  # en [m]
sigma= 450*10**6 #résitance à la compression en [Pa]
E= 200*10**9 #module d'élasticité en [pa]
Ix=(419/12)*t**4 # inertie du profil en "I" selon l'axe xx
#Iy=(131/12)*t**4 # inertie du profil en "I" selon l'axe yy
Kx=1 # flambage selon l'axe x (dans le plan du mouvement )
Ky= 0.5 # flambage selon l'axe y (perpendiculairement au mouvement )
A_I=12*t**2 #Aire de la surface de la bielle par rapport à x en [m^2]
L_b=5*t #longueur de la bielle en [m]
def flambage(theta):
    F_Euler=(np.pi)**2*E*Ix/(Kx*L_b)**2
    
    return (1/F_Euler+1/(A_I*sigma))**(-1)

def myfunc(rpm, s, theta, thetaC, deltaThetaC):
    #
    #!!!!!! theta en degrés et pas en radians
    #thetaC le décalage par rapport à zéro dans le sens négatif
    #
    theta = theta*pi/180
    thetaC = -thetaC*pi/180
    deltaThetaC = deltaThetaC*pi/180
    #
    #!!!!!! reconvertir le résultat final en fonction des degrés
    #
    return (V_output, Q_output, F_pied_output, F_tete_output, p_output, t)  #v
