import numpy as np 
import gym
from gym import spaces
import random as rd

class NetworkOperatorEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_cp, cache_capacity):
        
        rd.seed(5)
        self.liste_alpha=[0.65, 0.85, 0.95]
        self.liste_proba=[0.6, 0.30, 0.10]
        self.cons_zipf_1 = 8.13 
        self.cons_zipf_2 = 5.18
        self.cons_zipf_3 = 3.60
        self.conss_zipf = [self.cons_zipf_1, self.cons_zipf_2, self.cons_zipf_3]
        self.n_cp = n_cp
        self.cache_capacity = cache_capacity
        
        self.all_states = self._states_nCP(self.cache_capacity, self.n_cp)
        
        self.liste_nb_video = self.liste_100_videos(self.n_cp)
        
        self.rewards = []
        self.action_space = spaces.Discrete(7)
        
        # We associate an index to each state
        self.observation_space = spaces.Discrete(len(self.all_states))
        
        self.nb_requetes = 100
        self.current_r = 0
        self.nb_videos = 100
        self.current_step = 0
        self.max_steps = 100
    
    def _states_nCP(self, k,n):
        output_state_list=[]
        if n==1:
            output_state_list=[[n]]
        elif n==2:
            for j in range(k+1):
                output_state_list.append([j, k-j])
            return(output_state_list)
        else:
            for i in range (k+1):
                other_states = self._states_nCP(k-i,n-1)
                for state in other_states:
                    state.append(i)
                    output_state_list.append(state)
            return(output_state_list)
    
    def request_creation(self): #k
        somme=0
        i=0
        choix_CP = rd.random()
        while(True):
            somme = somme + self.liste_proba[i]
            if(choix_CP <= somme):
                break
            else:
                i +=1
        CP = i
        distribution = self._zipf_distribution(self.liste_alpha[i], self.liste_nb_video[i], self.conss_zipf[i]) #conss_zipf permet de soulages les calculs des constantes de normalisation de zipf
        choix_video = rd.random()
        compteur_choix = 0
        for j in range(1, self.liste_nb_video[i]+1):
            compteur_choix += distribution[j-1]
            if compteur_choix >= choix_video:
                video_choisie = j
                break   
        return [CP, video_choisie]
    
    def _zipf_distribution(self, alpha, nb_videos, norme): 
        """ Cette fonction permet de créer un input sur les vidéos d'un content provider
            Elle retourne le graphe des probabilotés pi de la vidéo i en fonction de i
            nb_videos est le nombre de films du catalogue du content provider
            alpha est le paramètre présent dans la loi de distribution de zipf"""
        norm = 0
        
        indices_videos = range(1, nb_videos+1)
        "indices_videos = range(1,nb_videos+1)" # necessaire pour tracer le plot (decocher si besoin)
        probabilites_pi = [0] * (nb_videos)
        for i in range(1, nb_videos+1):
            norm +=1.0/(i**alpha)
        for i in range(1, nb_videos+1):
            pi = (1.0/i**alpha) * (1.0/norm)
            probabilites_pi[i-1] = pi

        return probabilites_pi
    
    #Création liste 100 vidéo fix
    def liste_100_videos(self, k):
        l=[]
        for i in range(k):
            l.append(100)
        return l
    
    def _decide_opt_alloc(self, k, cache_capacity): # cache_capacity, k
        distribution=[0]*k
        popularite=[0]*k
        allocation=[0]*k
        pointeurs_max=[0]*k
        
        if cache_capacity > sum(self.liste_100_videos(k)):
            return 'Erreur: Le cache est trop grand par rapport au nombre total de videos'
        
        for i in range(k):
            distribution[i] = self._zipf_distribution(self.liste_alpha[i], self.liste_nb_video[i], self.conss_zipf[i])
            popularite[i] = [piyt * self.liste_proba[i] for piyt in distribution[i]] #liste que l'on va comparer
            
        for j in range (cache_capacity):
            max_temp=0 #popularite[0] par defaut
            for m in range(k-1):
                if popularite[m+1][pointeurs_max[m+1]]>popularite[m][pointeurs_max[m]]:
                    max_temp=m+1
            allocation[max_temp]=allocation[max_temp] + 1
            pointeurs_max[max_temp] = pointeurs_max[max_temp] + 1
            
        return allocation
        
    def reset(self):
        self.current_step = 0
        self.state = self._decide_opt_alloc(self.n_cp, self.cache_capacity)
        self.max_steps = 100
        
        return self._next_observation()
    
    def _next_observation(self):
        self.obs = self.observation_space.sample()
        self.state = self.all_states[self.obs]
        return self.obs
    
    # The action has already been chosen (it is action)
    # Now we just implement it
    def _take_action(self, action):
        
        if action == 1:
            self.state[0] += 1
            self.state[1] -= 1
        if action == 2:
            self.state[0] -= 1
            self.state[1] += 1
        if action == 3:
            self.state[0] += 1
            self.state[2] -= 1
        if action == 4:
            self.state[0] -= 1
            self.state[2] += 1
        if action == 5:
            self.state[1] += 1
            self.state[2] -= 1
        if action == 6:
            self.state[1] -= 1
            self.state[2] += 1
    
    def position_etat(self, alloc):
        compteur = -1
        for k in self.all_states:
            compteur +=1 ;
            if alloc == k:
                return compteur
    
    def step(self, action):
        self.current_step += 1
        done = False
        
        self._take_action(action)
        
        reward = 0
        
        for r in range(1, self.nb_requetes + 1):
            requete = self.request_creation()
            content_provider = requete[0]
            slots_allocated_to_that_content_provider = self.state[content_provider]
            video_id = requete[1]
            if slots_allocated_to_that_content_provider > video_id:
                reward +=1
        
        
        #reward -= self.nb_requetes
        #self.rewards.append(reward)
        
       
        """
        for cp in range(self.n_cp):
            requetes_vers_le_cp = self.nb_requetes * self.liste_proba[cp] 
            hit_ratio = sum(self._zipf_distribution(self.liste_alpha[cp], self.nb_videos, self.conss_zipf[cp])[0 : (self.state[cp]-1)])
            cout += requetes_vers_le_cp * (1- hit_ratio)
        """
        
        #if self.current_step >= self.max_steps:
        #    done = True
        
        obs = self._next_observation()
        
        return obs, reward, done, {}

    
    def render(self, mode='human', close=False):
    # Render the environment to the screen    
        plt.title('Evaluation' )
        plt.xlabel('Number of iterations')
        plt.ylabel('cost')
        plt.grid('on')
        plt.close()
        plt.show()