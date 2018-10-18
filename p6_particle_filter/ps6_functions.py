#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:30:10 2018

@author: jinzhao
"""
import cv2
import numpy as np

def img_show(images):
    if len(images) > 5:
        raise ValueError('A very specific bad thing happened.')
    for i in range(len(images)):
        img = np.uint8(images[i])
        cv2.imshow('img'+str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_frame(video_path, idx_frame):
    cap = cv2.VideoCapture(video_path)

    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if idx_frame >= n_frame:
        raise ValueError('Invalid frame index!')  
        
    count = 0
    while(True):
        ret, frame = cap.read()
        if count == idx_frame:
            return frame
        count += 1

def load_file(file_name):
    file = open(file_name, 'r')
    pos = np.float64(file.readline().strip().split())
    size = np.float64(file.readline().strip().split())
    file.close() 
    pos = pos.astype('int64')
    size = size.astype('int64')
    x_min, y_min = pos
    size_x, size_y = size
    return x_min, y_min, size_x, size_y

def draw_bounding_box(frame, x_start, y_start, size_x, size_y, show=False):
    frame_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if show:
        img_show([frame_draw])
    
    return frame_draw

def visualize_frame_naive(frame, model, opt_state):
    size_y, size_x = model.shape
    x_start = opt_state[0] - size_x//2
    y_start = opt_state[1] - size_y//2
    
    cv2.rectangle(frame, (x_start, y_start), (x_start+size_x, y_start+size_y), (0, 255, 0), 3) # UL corner -- DR corner
    cv2.circle(frame, (opt_state[0], opt_state[1]), 2, (0, 0, 255), -1)

    return frame

def visualize_frame(frame, S, model, opt_state, std):
    #1. Draw particles
    for particle in S:
        cv2.circle(frame, (particle.state[0], particle.state[1]), 1, (255, 0, 0), -1)
        
    #2. Draw the bounding box with weighted sum (optimal) particle
    size_y, size_x = model.shape
    x_start = opt_state[0] - size_x//2
    y_start = opt_state[1] - size_y//2
    
    cv2.circle(frame, (opt_state[0], opt_state[1]), 2, (0, 0, 255), -1)
    cv2.rectangle(frame, (x_start, y_start), (x_start+size_x, y_start+size_y), (0, 255, 0), 3) # UL corner -- DR corner
    
    #3. Draw the spread of the distribution 
    cv2.circle(frame, (opt_state[0], opt_state[1]), int(std), (255, 255, 255), 0)

    return frame

    

# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
class Particle:
    def __init__(self, state, weight):
        self.state = state
        self.weight = weight
    def normalize_weight(self, sum_weight):
        self.weight /= sum_weight
        
class ParticleTracker:
    def __init__(self, model, sample_space, std_control=10, n_particles=100, std_MSE=20, alpha = 0):
        self.model = model 
        self.sample_space = sample_space
        
        self.std_control = std_control
        self.n_particles = n_particles
        self.std_MSE = std_MSE
        self.alpha = alpha

        self.init_particles()
                
    def init_particles(self):
        self.S = [0] * self.n_particles
        x_min, y_min, x_max, y_max = self.sample_space # Caution the order
        for i in range(self.n_particles):
            state = np.array([[np.random.uniform(x_min, x_max)], \
                     [np.random.uniform(y_min, y_max)]])
            weight = 1 / self.n_particles
            particle = Particle(state, weight)
            self.S[i] = particle
            
    def update(self, frame): # refer to the slide
        S_tmp = [0] * self.n_particles # will store the updated particle temporarily
        sum_weight = 0
        for i in range(self.n_particles):
            idx_particle = self.particle_resample()
            state = self.particle_control(idx_particle, frame)
            weight = self.compute_importance_weight(state, frame)
            sum_weight += weight
            S_tmp[i] = Particle(state, weight)
        for i in range(self.n_particles):
            particle = S_tmp[i]
            particle.normalize_weight(sum_weight) # Keep state but normalize the weight
            self.S[i] = particle
        if self.alpha != 0: # This condition is trivial. I put here just for future reference
            self.update_appearance_model(frame)
    def particle_resample(self):
        # Roulette wheel selection
        rand_weight = np.random.random_sample() # Return random floats in [0.0, 1.0)
        sum_weight = 0
        for idx in range(self.n_particles):
            particle = self.S[idx]
            sum_weight += particle.weight
            if sum_weight > rand_weight:
                return idx
    
    def particle_control(self, idx_particle, frame):
        noise = np.random.normal(0, self.std_control, (2, 1))
        particle = self.S[idx_particle]
        state = particle.state + noise
        
        state[0, 0] = np.clip(state[0, 0], 0, frame.shape[1])
        state[1, 0] = np.clip(state[1, 0], 0, frame.shape[0])
        state = state.astype('int64')

        return state
    
    def compute_importance_weight(self, state, frame):
        size_y, size_x = self.model.shape
     
        x_start = state[0, 0]-size_x//2
        y_start = state[1, 0]-size_y//2
        x_end = x_start + size_x
        y_end = y_start + size_y

        particle_patch = frame[y_start : y_end, x_start : x_end]
        
        if particle_patch.shape != self.model.shape:
            return 0
        else:
            MSE = np.mean((self.model - particle_patch)**2)
            weight = np.exp( - MSE / (2*self.std_MSE**2) )
            return weight
    
    def update_appearance_model(self, frame):
        size_y, size_x = self.model.shape
        #best_state = self.determine_weightedSum_state()
        best_state = self.determine_best_state()
             
        x_start = best_state[0, 0]-size_x//2
        y_start = best_state[1, 0]-size_y//2
        x_end = x_start + size_x
        y_end = y_start + size_y

        cur_best_model = frame[y_start : y_end, x_start : x_end]
        if cur_best_model.shape == self.model.shape:
             self.model = self.alpha * cur_best_model + (1-self.alpha) * self.model
        
    def determine_best_state(self):
        weights_set = [0] * self.n_particles
        for i in range(self.n_particles):
            weights_set[i] = self.S[i].weight

        best_idx = np.argmax(weights_set)

        best_state = self.S[best_idx].state
        
        return best_state
    
    def determine_weightedSum_state(self):
        best_state = np.zeros((2, 1))
        for i in range(self.n_particles):
            best_state += self.S[i].state * self.S[i].weight
        best_state = best_state.astype('int64')
        
        return best_state
    
    def compute_spreadOfDistribution(self):
        opt_state = self.determine_weightedSum_state()
        std = 0
        for i in range(self.n_particles):
            std += self.S[i].weight * np.linalg.norm(self.S[i].state - opt_state)
        
        return std
    
    def visualize_frame(self, frame):
        #opt_state = self.determine_best_state()
        opt_state = self.determine_weightedSum_state()
        std = self.compute_spreadOfDistribution()
        #1. Draw particles
        for particle in self.S:
            cv2.circle(frame, (particle.state[0], particle.state[1]), 1, (255, 0, 0), -1)
            
        #2. Draw the bounding box with weighted sum (optimal) particle
        size_y, size_x = self.model.shape
        x_start = opt_state[0] - size_x//2
        y_start = opt_state[1] - size_y//2
        
        cv2.circle(frame, (opt_state[0], opt_state[1]), 2, (0, 0, 255), -1)
        cv2.rectangle(frame, (x_start, y_start), (x_start+size_x, y_start+size_y), (0, 255, 0), 3) # UL corner -- DR corner
        
        #3. Draw the spread of the distribution 
        cv2.circle(frame, (opt_state[0], opt_state[1]), int(std), (255, 255, 255), 0)
    
        return frame

# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 
def compare_hist(img1, img2, n_bins=8): # img format must be either 'uint8' or 'float32'
    hist1 = np.zeros((1, 3*n_bins)).astype('float32')
    hist2 = np.zeros((1, 3*n_bins)).astype('float32')
    
    for i in range(3):
        hist1[0, i*n_bins : i*n_bins + n_bins] = cv2.calcHist([img1], [i], None, [n_bins], [0, 256]).T
        hist2[0, i*n_bins : i*n_bins + n_bins] = cv2.calcHist([img2], [i], None, [n_bins], [0, 256]).T
        
        hist1[0, i*n_bins : i*n_bins+n_bins] /= hist1[0, i*n_bins:i*n_bins+n_bins].sum()
        hist2[0, i*n_bins : i*n_bins+n_bins] /= hist2[0, i*n_bins:i*n_bins+n_bins].sum()
    
    # Chi-Squared distance
    weight = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        
    return weight

class ParticleTracker_MSL:
    def __init__(self, model, sample_space, std_control=5, n_particles=200, std=1, alpha=0):
        self.model = model 
        self.sample_space = sample_space
        
        self.std_control = std_control
        self.n_particles = n_particles
        self.std = std
        self.alpha = alpha

        self.init_particles()
                
    def init_particles(self):
        self.S = [0] * self.n_particles
        x_min, y_min, x_max, y_max = self.sample_space # Mind the order
        for i in range(self.n_particles):
            state = np.array([[np.random.uniform(x_min, x_max)], \
                     [np.random.uniform(y_min, y_max)]])
            weight = 1 / self.n_particles
            particle = Particle(state, weight)
            self.S[i] = particle
            
    def update(self, frame): # refer to the slide
        S_tmp = [0] * self.n_particles # will store the updated particle temporarily
        sum_weight = 0
        for i in range(self.n_particles):
            idx_particle = self.particle_resample()
            state = self.particle_control(idx_particle, frame)
            weight = self.compute_importance_weight(state, frame)
            sum_weight += weight
            S_tmp[i] = Particle(state, weight)
        for i in range(self.n_particles):
            particle = S_tmp[i]
            particle.normalize_weight(sum_weight) # Keep state but normalize the weight
            self.S[i] = particle
        if self.alpha != 0: # This condition is trivial. I put here just for future reference
            self.update_appearance_model(frame)
            
    def particle_resample(self):
        # Roulette wheel selection
        rand_weight = np.random.random_sample() # Return random floats in [0.0, 1.0)
        sum_weight = 0
        for idx in range(self.n_particles):
            particle = self.S[idx]
            sum_weight += particle.weight
            if sum_weight > rand_weight:
                return idx
    
    def particle_control(self, idx_particle, frame):
        noise = np.random.normal(0, self.std_control, (2, 1))
        particle = self.S[idx_particle]
        state = particle.state + noise
        
        state[0, 0] = np.clip(state[0, 0], 0, frame.shape[1])
        state[1, 0] = np.clip(state[1, 0], 0, frame.shape[0])
        state = state.astype('int64')

        return state
    
    def compute_importance_weight(self, state, frame):
        size_y, size_x, _ = self.model.shape
     
        x_start = state[0, 0]-size_x//2
        y_start = state[1, 0]-size_y//2
        x_end = x_start + size_x
        y_end = y_start + size_y

        particle_patch = frame[y_start : y_end, x_start : x_end, :]
        
        if particle_patch.shape != self.model.shape:
            weight = 0
        else:
            dist = compare_hist(self.model, particle_patch)
            weight = np.exp( - dist / (2 * self.std**2) )
            
        return weight
    
    def update_appearance_model(self, frame):
        size_y, size_x, _ = self.model.shape
        #best_state = self.determine_weightedSum_state()
        best_state = self.determine_best_state()
             
        x_start = best_state[0, 0]-size_x//2
        y_start = best_state[1, 0]-size_y//2
        x_end = x_start + size_x
        y_end = y_start + size_y

        cur_best_model = frame[y_start : y_end, x_start : x_end]
        if cur_best_model.shape == self.model.shape:
             self.model = self.alpha * cur_best_model + (1-self.alpha) * self.model
        
    def determine_best_state(self):
        weights_set = [0] * self.n_particles
        for i in range(self.n_particles):
            weights_set[i] = self.S[i].weight

        best_idx = np.argmax(weights_set)

        best_state = self.S[best_idx].state
        
        return best_state
    
    def determine_weightedSum_state(self):
        best_state = np.zeros((2, 1))
        for i in range(self.n_particles):
            best_state += self.S[i].state * self.S[i].weight
        best_state = best_state.astype('int64')
        
        return best_state
    
    def compute_spreadOfDistribution(self):
        opt_state = self.determine_weightedSum_state()
        std = 0
        for i in range(self.n_particles):
            std += self.S[i].weight * np.linalg.norm(self.S[i].state - opt_state)
        
        return std
    def visualize_frame(self, frame):
        #opt_state = self.determine_best_state()
        opt_state = self.determine_weightedSum_state()
        std = self.compute_spreadOfDistribution()
        
        #1. Draw particles
        for particle in self.S:
            cv2.circle(frame, (particle.state[0], particle.state[1]), 1, (255, 0, 0), -1)
            
        #2. Draw the bounding box with weighted sum (optimal) particle
        size_y, size_x, _ = self.model.shape
        x_start = opt_state[0] - size_x//2
        y_start = opt_state[1] - size_y//2
        
        cv2.circle(frame, (opt_state[0], opt_state[1]), 2, (0, 0, 255), -1)
        cv2.rectangle(frame, (x_start, y_start), (x_start+size_x, y_start+size_y), (0, 255, 0), 3) # UL corner -- DR corner
        
        #3. Draw the spread of the distribution 
        cv2.circle(frame, (opt_state[0], opt_state[1]), int(std), (255, 255, 255), 0)
    
        return frame


    
        