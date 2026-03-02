"""
Emotion Classification with Brain Circuit Simulation
Fear Amygdala Model

This module simulates the amygdala's response to threat stimuli.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import csv
from dataclasses import dataclass


@dataclass
class SimulationParameters:
    stimulus_intensity: float = 0.3
    firing_threshold: float = 0.5
    neural_gain: float = 1.0
    simulation_duration: float = 20.0
    time_step: float = 0.01
    noise_level: float = 0.05
    inhibition_strength: float = 0.3


class AmygdalaModel:
    """Amygdala neural circuit model for fear response simulation."""
    
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.membrane_potential = 0.0
        self.firing_rate = 0.0
        self.time = 0.0
        
        self.time_history = []
        self.potential_history = []
        self.firing_rate_history = []
        self.stimulus_history = []
        
    def _get_stimulus(self, t):
        """Generate Gaussian threat stimulus"""
        stimulus_onset = self.params.simulation_duration * 0.2
        stimulus_width = 2.0
        return self.params.stimulus_intensity * np.exp(-((t - stimulus_onset) ** 2) / (2 * stimulus_width ** 2))
    
    def _neuronal_dynamics(self, V, stimulus):
        """Neural dynamics: dV/dt = -V + stimulus - inhibition + noise"""
        noise = np.random.normal(0, self.params.noise_level)
        dV = -V + (stimulus * self.params.neural_gain) - (self.firing_rate * self.params.inhibition_strength) + noise
        return dV
    
    def _update_firing_rate(self, V):
        """F-I curve: sigmoid activation"""
        if V > self.params.firing_threshold:
            return 1.0 / (1.0 + np.exp(-5 * (V - self.params.firing_threshold)))
        return 0.0
    
    def step(self, stimulus):
        """Euler integration step"""
        dV = self._neuronal_dynamics(self.membrane_potential, stimulus)
        self.membrane_potential += dV * self.params.time_step
        self.membrane_potential = np.clip(self.membrane_potential, -1.0, 1.0)
        
        self.firing_rate = self._update_firing_rate(self.membrane_potential)
        
        self.time_history.append(self.time)
        self.potential_history.append(self.membrane_potential)
        self.firing_rate_history.append(self.firing_rate)
        self.stimulus_history.append(stimulus)
        
        self.time += self.params.time_step
    
    def run(self):
        """Run simulation"""
        num_steps = int(self.params.simulation_duration / self.params.time_step)
        print(f"Running simulation for {self.params.simulation_duration}s...")
        
        for step in range(num_steps):
            stimulus = self._get_stimulus(self.time)
            self.step(stimulus)
            
            if (step + 1) % int(num_steps / 10) == 0:
                print(f"  {((step + 1) / num_steps) * 100:.0f}%")
        
        print("Done!")
    
    def get_oscillation_frequency(self):
        """Calculate dominant oscillation frequency"""
        firing_array = np.array(self.firing_rate_history)
        threshold = np.mean(firing_array[int(len(firing_array)/2):]) * 0.8
        
        crossings = np.where(np.diff(firing_array > threshold).astype(int) == 1)[0]
        
        if len(crossings) < 2:
            return 0.0
        
        intervals = np.diff(crossings)
        avg_interval = np.mean(intervals) * self.params.time_step
        
        return 1.0 / (2 * avg_interval) if avg_interval > 0 else 0.0
    
    def classify_fear_state(self):
        """Classify brain state as FEAR or NO FEAR"""
        steady_state = np.array(self.firing_rate_history[int(len(self.firing_rate_history)/2):])
        mean_firing = np.mean(steady_state)
        oscillation_freq = self.get_oscillation_frequency()
        
        if oscillation_freq > 15.0 or mean_firing > 0.7:
            state = "FEAR"
            confidence = min(oscillation_freq / 40.0, 1.0)
        else:
            state = "NO FEAR"
            confidence = 1.0 - min(oscillation_freq / 15.0, 1.0)
        
        return {
            "state": state,
            "confidence": confidence,
            "oscillation_frequency": oscillation_freq,
            "mean_firing_rate": mean_firing
        }
    
    def plot_results(self, filename="amygdala_plot.png"):
        """Plot simulation results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        time = np.array(self.time_history)
        potential = np.array(self.potential_history)
        firing = np.array(self.firing_rate_history)
        stimulus = np.array(self.stimulus_history)
        
        axes[0].plot(time, stimulus, 'b-', linewidth=2, label='Threat Stimulus')
        axes[0].set_ylabel('Stimulus Intensity')
        axes[0].set_title('Amygdala Fear Response Simulation')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(time, potential, 'g-', linewidth=1.5, label='Membrane Potential')
        axes[1].axhline(y=self.params.firing_threshold, color='r', linestyle='--', alpha=0.7, label='Firing Threshold')
        axes[1].set_ylabel('Membrane Potential (V)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].plot(time, firing, 'r-', linewidth=1.5, label='Neural Firing Rate')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Firing Rate')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
    
    def save_data_csv(self, filename="amygdala_data.csv"):
        """Save data to CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)', 'Stimulus', 'Membrane Potential', 'Firing Rate'])
            for t, stim, pot, firing in zip(self.time_history, self.stimulus_history, 
                                            self.potential_history, self.firing_rate_history):
                writer.writerow([t, stim, pot, firing])
        print(f"Data saved to {filename}")
    
    def print_summary(self):
        """Print results summary"""
        fear = self.classify_fear_state()
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Stimulus Intensity: {self.params.stimulus_intensity}")
        print(f"Firing Threshold: {self.params.firing_threshold}")
        print(f"\nBrain State: {fear['state']}")
        print(f"Confidence: {fear['confidence']:.2%}")
        print(f"Oscillation Frequency: {fear['oscillation_frequency']:.2f} Hz")
        print(f"Mean Firing Rate: {fear['mean_firing_rate']:.3f}")
        print("="*60 + "\n")


def load_config(config_file="config.json"):
    """Load parameters from config file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return SimulationParameters(**config)
    except FileNotFoundError:
        return SimulationParameters()


def main():
    parser = argparse.ArgumentParser(description="Amygdala Fear Response Simulation")
    parser.add_argument('--stimulus-intensity', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--neural-gain', type=float, default=None)
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--no-csv', action='store_true')
    
    args = parser.parse_args()
    
    params = load_config(args.config)
    
    if args.stimulus_intensity is not None:
        params.stimulus_intensity = args.stimulus_intensity
    if args.threshold is not None:
        params.firing_threshold = args.threshold
    if args.neural_gain is not None:
        params.neural_gain = args.neural_gain
    if args.duration is not None:
        params.simulation_duration = args.duration
    
    model = AmygdalaModel(params)
    model.run()
    model.print_summary()
    
    if not args.no_plot:
        model.plot_results()
    
    if not args.no_csv:
        model.save_data_csv()


if __name__ == "__main__":
    main()
