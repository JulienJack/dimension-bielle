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
        theta = np.copy(theta)
        theta[(thetaC > theta) | (theta > thetaC + deltaThetaC)] = thetaC
    elif (thetaC > theta) or (theta > thetaC + deltaThetaC) : return 0.0
    return Q*(1-cos(pi*(theta-thetaC)/deltaThetaC))/2 # en [J]

def derivee_chaleur(theta):
    if type(theta) != np.float64 and type(theta) != type(0.0) :
        theta = np.copy(theta)
        theta[(thetaC > theta) | (theta > thetaC + deltaThetaC)] = thetaC
    elif (thetaC > theta) or (theta > thetaC + deltaThetaC) : return 0.0
    return Q*pi*sin(pi*(theta-thetaC)/deltaThetaC)/(2*deltaThetaC)

def derivee_pression(Theta,p):
    return ( -gamma*p*derivee_volume(Theta) + (gamma-1)*derivee_chaleur(Theta) )/volume(Theta)

def forces_bielle(theta,rpm): #bilan des forces
    omega= rpm*(2*np.pi)/60  # vitesse angulaire PAR RAPPORT à l'axe TOURILLON ? en [rad/s]
    p = pression_cylindre(theta)
    F_pied_output=(np.pi*D**2)*p/4-mpiston*R*cos(theta)*(omega)**2
    F_tete_output=-(np.pi*D**2)*p/4+(mpiston+mbielle)*R*cos(theta)*(omega)**2
    return (F_pied_output,F_tete_output,F_pied_output-F_tete_output) # La compression de la bielle cad la somme de la force appliquée sur la tete et sur le pied de la bielle en [N]


def pression_cylindre(theta):
    result = solve_ivp(derivee_pression,(-pi,pi),np.array([s*10**5]),t_eval = theta)
    #if(result.t != theta) : print("Attention aux angles évalués")
    return result.y.reshape(len(result.y[0])) # en [Pa]

def epaisseur_critique(theta): #rend la dimension de t limite par rapport au flambage en x
    t=(((((force_bielle(theta))**-1)-1/(np.pi)**2*E*Ix/(Kx*L_b)**2)**-1)/sigma*11)**(1/2)
    return t  # taille minimale de t, mesure caractérisant le dimensionnement de la bielle [m]

sigma= 450*10**6 #résitance à la compression en [Pa]
E= 200*10**9 #module d'élasticité en [pa]
Ix=(419/12)*t**4 # inertie du profil en "I" selon l'axe xx
#Iy=(131/12)*t**4 # inertie du profil en "I" selon l'axe yy
Kx=1 # flambage selon l'axe x (dans le plan du mouvement )
Ky= 0.5 # flambage selon l'axe y (perpendiculairement au mouvement )
A_I=11*t**2 #Aire de la surface de la bielle par rapport à x en [m^2]
L_b=L # ??? ??? ???? 


def flambage(theta): # rend la force critique que lon peut exercer sur la bielle avant un flambage en x et y
    F_Eulerx=(np.pi)**2*E*Ix/(Kx*L_b)**2
    F_Eulery=(np.pi)**2*E*Iy/(Ky*L_b)**2
    return (1/F_Eulerx+(1/(A_I*sigma)))**(-1),(1/F_Eulery+(1/(A_I*sigma)))**(-1)
#Force critique à appliquer avant le flambage de la bielle en [N]

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



############################### PROGRAMME MODIFIE ##############################################

import numpy as np
from numpy import sqrt,cos,sin,pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

tau = 10 #valeur taux compression# #[-] = Volume au point mort bas PMB (vol max) / volume au point mort haut PMH (vol min)
# pour l essence le taux vaire de 1.0 a 1.2 MPa et pour le diesel de 3.0 a 3.5 MPa
D = 0.15 #valeur alesage# #[m] = diametre du cylindre dans lequel le piston rentre
C = 0.20 #valeur course# #[m] = la course est le déplacement du piston donc max de 2R
Vc = pi*((D)**2)*C/4 #[-] = la cylindrée, soit le volume max - volume min

L = 0.25 #valeur longueur bielle#  # [m] = la tige entre le piston et le vilebrequin ( relié au maneton du vilebrequin )
mpiston = 0.5 #valeur masse piston# #[kg] =
mbielle = 0.4 #valeur masse bielle# #[kg] = connecting rod
Q = 2800000 #valeur chaleur emise par fuel par kg de melange admis# #[J/kg_inlet gas]
gamma = 1.3 #coefficient isentropique

#Deux directions de flambages ?

R=C/2 #distance entre le maneton et le tourillon# #en [m]
b=L/R

#theta en degre ss forme d un vecteur de - 180 a 180
#Mais on va simuler un cycle complet ? donc 2* 360 ?
theta= np.arange(-180,180+0.5,1)
H=np.zeros(len(theta))
V_output=np.zeros(len(theta))

thetaC = 10
deltaThetaC = 15
s = 1

sigma_c = 450*10**6
Kx = 1    #ATTENTION : prendre la valeur donnée par l'enoncée ou la valeur recommandée (recommanded design value)
Ky = 0.5
E = 200*10**9
coeff_x = 419/12
coeff_y = 131/12



class Moteur :
    
    def __init__(self,L,R,D,mpiston,mbielle,tau,s,rpm,thetaC,deltaThetaC,Q,Tadm):
        self.L = L
        self.R = R
        self.b = L/R
        self.D = D
        self.Vc = pi*R*D**2/2
        self.mpiston = mpiston
        self.mbielle = mbielle
        self.tau = tau
        self.s = s
        self.rpm = rpm
        self.thetaC = thetaC
        self.deltaThetaC = deltaThetaC
        self.Qtot = Q*self.Vc*tau/(tau-1)*s*10**5/(287*(Tadm+273.15))
    
    
    def volume(self,theta):
        return (self.Vc/2)*(1-cos(theta)+self.b-sqrt(self.b**2-sin(theta)**2))+(self.Vc)/(self.tau-1)


    def derivee_volume(self,theta):
        return (self.Vc/2)*(sin(theta)+(sin(theta)*cos(theta))/(sqrt(self.b**2-sin(theta)**2)))


    def apport_chaleur(self,theta): # calculer l'apport de chaleur sur la durée de temps de combustion voir son schéma
        if type(theta) != np.float64 and type(theta) != type(0.0) :
            theta = np.copy(theta)
            theta[(self.thetaC > theta) | (theta > self.thetaC + self.deltaThetaC)] = self.thetaC
        elif (self.thetaC > theta) or (theta > self.thetaC + self.deltaThetaC) : return 0.0
        return self.Qtot*(1-cos(pi*(theta-self.thetaC)/self.deltaThetaC))/2 # en [J]


    def derivee_chaleur(self,theta):
        if type(theta) != np.float64 and type(theta) != type(0.0) :
            theta = np.copy(theta)
            theta[(self.thetaC > theta) | (theta > self.thetaC + self.deltaThetaC)] = self.thetaC
        elif (self.thetaC > theta) or (theta > self.thetaC + self.deltaThetaC) : return 0.0
        return self.Qtot*pi*sin(pi*(theta-self.thetaC)/self.deltaThetaC)/(2*self.deltaThetaC)


    def derivee_pression(self,Theta,p):
        return ( -gamma*p*self.derivee_volume(Theta) + (gamma-1)*self.derivee_chaleur(Theta) )/self.volume(Theta)


    def forces_bielle(self,theta,p = None): #bilan des forces
        if p is None :
            p = self.pression_cylindre(theta)
        omega = self.rpm*(2*pi)/60  # vitesse angulaire PAR RAPPORT à l'axe TOURILLON ? en [rad/s]
        F_pied_output = (pi*self.D**2)*p/4-self.mpiston*self.R*cos(theta)*(omega)**2
        F_tete_output = -(pi*self.D**2)*p/4+(self.mpiston+self.mbielle)*self.R*cos(theta)*(omega)**2
        return (F_pied_output,F_tete_output,F_pied_output-F_tete_output) # La compression de la bielle cad la somme de la force appliquée sur la tete et sur le pied de la bielle en [N]


    def pression_cylindre(self,theta):
        result = solve_ivp(self.derivee_pression,(-pi,pi),np.array([self.s*10**5]),t_eval = theta)
        return result.y.reshape(len(result.y[0])) # en [Pa]


    def epaisseur_critique(self,theta,f_crit = None, p = None): #? voir schema de l enonce
        if f_crit is None :
            f_crit = max(self.forces_bielle(theta,p)[2])
        delta_ux = 1/(121*sigma_c**2) + 4*(Kx*self.L)**2/(f_crit*pi**2*E*coeff_x)
        delta_uy = 1/(121*sigma_c**2) + 4*(Ky*self.L)**2/(f_crit*pi**2*E*coeff_y)
        t = sqrt(f_crit*( 1/(11*sigma_c) + sqrt(max(delta_ux,delta_uy)) )/2)
        return t  # en [m]




def myfunc(rpm, s, theta, thetaC, deltaThetaC):
    #
    #!!!!!! theta en degrés et pas en radians
    #thetaC le décalage par rapport à zéro dans le sens négatif
    #
    print(theta)
    theta = np.copy(theta)*pi/180
    print(theta)
    thetaC = -thetaC*pi/180
    deltaThetaC = deltaThetaC*pi/180
    #
    #!!!!!! reconvertir le résultat final en fonction des degrés
    #
    M = Moteur(L,R,D,mpiston,mbielle,tau,s,rpm,thetaC,deltaThetaC,Q,30)
    V_output = M.volume(theta)
    Q_output = M.apport_chaleur(theta)
    p_output = M.pression_cylindre(theta)
    (F_pied_output,F_tete_output,F_compress) = M.forces_bielle(theta, p_output)
    t = M.epaisseur_critique(theta, f_crit = max(F_compress))
    return (V_output, Q_output, F_pied_output, F_tete_output, p_output, t)  #v




