import numpy as np
import matplotlib.pyplot as plt

### Defining the acceptance probability function based on centrality percentile
# The calibration loader reads the precomputed entropy-to-centrality mapping from a file for each design point.
# The acceptance probability is then defined as a function of centrality percentile.

class AcceptancePolicy:
    """
    Defines acceptance probability P_acc(E) based on Centrality Percentile C(E)
    """
    def __init__(self, calibrator, uc_cutoff_pct: float = 1.0, transition_pct: 10.0, min_p_acc: float = 0.05):
        """
        Initializes the AcceptancePolicy with a given centrality percentile.

        Parameters:
        centrality_percentile (float): The centrality percentile C(E) for acceptance probability.
        """
        self.calibrator = calibrator
        self.uc_cutoff = uc_cutoff_pct
        self.transition = transition_pct
        self.min_p_acc = min_p_acc

    def acceptance_probability(self, E_val: float) -> float: # is not better to use centrality_pct as input?
        """
        Computes the acceptance probability P_acc(E) based on the centrality percentile C(E).
        """
        c = self.calibrator.E_to_centrality(E_val)

        if c <= self.uc_cutoff:
            return 1.0
        elif c <= self.transition:
            # Linear ramp-down from 1.0 to min_p_acc
            slope = (self.min_p_acc - 1.0) / (self.transition - self.uc_cutoff)
            return 1.0 + slope * (c - self.uc_cutoff)
        else:
            return self.min_p_acc

        ### Add more options for different acceptance policies if needed in the future.
        #  step-function, exponential decay, etc. can be implemented here.    
        
### Event filter & hydro triggering based on acceptance probability. 
# Decide online whether to pass an event to Hydro or discard it.    
class EventFilterPipeline:
    """
    Event filter pipeline that decides whether to pass an event to Hydro or discard it based on acceptance probability.
    """
    def __init__(self, acceptance_policy: AcceptancePolicy):
        self.calibrator = TrentoCalibrator.load(calibrator_path)
        self.policy = AcceptancePolicy(self.calibrator, uc_cutoff_pct, transition_pct=10.0, min_p_acc)

    def process_trento_event(self, event_id: int, total_entropy: float):
        """
        Evaluates a single Trento event and determines if it should proceed to Hydro 
        
        Returns:
        --------
        dict : Status containing acceptance flag, centrality, and statistical weight.
        """
        centrality = self.calibrator.E_to_centrality(total_entropy)
        p_acc = self.policy.acceptance_probability(centrality)

        # Acceptance decision via Monte Carlo rejection 
        rng = np.random.default_rng(seed=event_id + 123456789)  # Use a random number generator for reproducibility (save this probability)
        accepted = rng.random() < p_acc

        weight = 1.0 / p_acc if accepted else 0.0 # statistical weight for accepted events

        return {
            "event_id": event_id,
            "total_entropy": total_entropy,
            "centrality_pct": centrality,
            "p_acc": p_acc,
            "accepted": accepted,
            "weight": weight        
        }
if __name__ == "__main__":
    # Example usage
    pipeline = EventFilterPipeline("trento_calibration.dp0.pkl", uc_cutoff_pct=1.0, transition_pct=10.0, min_p_acc=0.05)
    
    # Simulate processing incoming TRENTo events
    print("\n--- Testing Online Event Filtering ---")
    np.random.seed(100)
    test_events_E = np.random.gamma(shape=15.0, scale=35.0, size=10)  # Simulated total entropy values for 10 events

    for i, E_val in enumerate(test_events_E):
        res = pipeline.process_trento_event(event_id=i, total_entropy=E_val)
        status = "KEEP -> Pass to Hydro" if res['accepted'] else "REJECT -> Discard"
        print(f"Event {i:02d} | E={E_val:6.1f} | Centrality={res['centrality_pct']:5.2f}% | P_acc={res['p_acc']:.3f} | {status} (Weight={res['weight']:.2f})")


### Weighted Observables Calculation
# Calculate unbiased physical observables from the accepted events using their statistical weights.

def effective_size(observables: np.ndarray, weights: np.ndarray):
    """
    Computes the effective sample size of observables based on their statistical weights.

    Parameters:
    -----------
    observables : np.ndarray
        Array of observable values from accepted events.
    weights : np.ndarray
        Array of statistical weights corresponding to the observables.

    Returns:
    --------
    N_eff : Effective sample size based on Kish's approximation.
    """
    W_sum = np.sum(weights)
    W2_sum = np.sum(weights**2)

    # Effective sample size (Kish's approximation)
    N_eff = (W_sum**2) / W2_sum if W2_sum > 0 else 0.0

    return N_eff


### Plotting and validations
# Generate 1 million minimum bias events from Trento
# Plot the histogram of total entropy and centrality percentile

def plot_event_distributions(events: List[dict]):
    """
    Plots the distributions of total entropy and centrality percentile for a list of events.
    """
    total_entropies = [event['total_entropy'] for event in events]
    centralities = [event['centrality_pct'] for event in events]

    plt.figure(figsize=(12, 5))

    # Histogram of total entropy
    plt.subplot(1, 2, 1)
    plt.hist(total_entropies, bins=50, color='blue', alpha=0.7)
    plt.title('Total Entropy Distribution')
    plt.xlabel('Total Entropy')
    plt.ylabel('Number of Events')

    # Histogram of centrality percentiles
    plt.subplot(1, 2, 2)
    plt.hist(centralities, bins=50, color='green', alpha=0.7)
    plt.title('Centrality Percentile Distribution')
    plt.xlabel('Centrality Percentile')
    plt.ylabel('Number of Events')

    plt.tight_layout()
    plt.show()

# Recover the original minimum bias distribution of total entropy and centrality percentile from the accepted events using their statistical weights.

def recover_min_bias_distribution(accepted_events: List[dict]):
    """
    Recovers the original minimum bias distribution of total entropy and centrality percentile
    from the accepted events using their statistical weights.
    """
    total_entropy_weights = np.array([event['total_entropy'] * event['weight'] for event in accepted_events])
    centrality_weights = np.array([event['centrality_pct'] * event['weight'] for event in accepted_events])
    weights = np.array([event['weight'] for event in accepted_events])

    # Compute the effective distributions
    min_bias_total_entropy = total_entropy_weights / weights.sum() if weights.sum() > 0 else 0
    min_bias_centrality = centrality_weights / weights.sum() if weights.sum() > 0 else 0

    return min_bias_total_entropy, min_bias_centrality

# Try to calculate the level of synchronization per design point based on the accepted events and their statistical weights. 
# This can be useful for understanding how well the accepted events represent the original distribution.
# Here, we must print the events_id numbers to see which events were accepted and which were rejected, along with their weights for each design point.
# At the end, we can plot the synchronization level for each design point to visualize how well the accepted events represent the original distribution.
def calculate_synchronization_level(accepted_events: List[dict]):
    """
    Calculates the level of synchronization per design point based on the accepted events and their statistical weights.
    """
    # Group events by design point
    design_points = {}
    for event in accepted_events:
        dp = event['design_point']
        if dp not in design_points:
            design_points[dp] = []
        design_points[dp].append(event)

    # Compute synchronization level for each design point
    sync_levels = {}
    for dp, events in design_points.items():
        # Recover minimum bias distribution for this design point
        min_bias_total_entropy, min_bias_centrality = recover_min_bias_distribution(events)
        sync_levels[dp] = {
            'min_bias_total_entropy': min_bias_total_entropy,
            'min_bias_centrality': min_bias_centrality
        }
    
    # print the events_id numbers to see which events were accepted and which were rejected, along with their weights for each design point
    for dp, events in design_points.items():
        print(f"\nDesign Point: {dp}")
        for event in events:
            status = "ACCEPTED" if event['accepted'] else "REJECTED"
            print(f"Event ID: {event['event_id']} | Status: {status} | Weight: {event['weight']:.2f}") 
    return sync_levels

# We need to find a well-defined test case to evaluate the Python random number generator across different platforms and Python versions to ensure reproducibility of the acceptance decision.

# Another thing that I want to plot is the acceptance probability as a function of centrality percentile for different design points to visualize how the acceptance policy behaves across the range of centrality percentiles. 
# This can help in understanding the impact of the acceptance policy on the event selection process.
def plot_acceptance_probability(synchronization_levels: dict):
    """
    Plots the acceptance probability as a function of centrality percentile for different design points.
    """
    plt.figure(figsize=(10, 6))
    for dp, levels in synchronization_levels.items():
        plt.plot(levels['min_bias_centrality'], label=f"DP {dp}")

    plt.title('Acceptance Probability vs Centrality Percentile')
    plt.xlabel('Centrality Percentile')
    plt.ylabel('Acceptance Probability')
    plt.legend()
    plt.grid()
    plt.show()

# Plot the effective sample size divided by the total number of events for each design point to visualize how well the accepted events represent the original distribution.
def plot_effective_sample_size(synchronization_levels: dict):
    """
    Plots the effective sample size divided by the total number of events for each design point.
    """
    plt.figure(figsize=(10, 6))
    for dp, levels in synchronization_levels.items():
        N_eff = effective_size(levels['min_bias_total_entropy'], levels['min_bias_centrality'])
        plt.bar(dp, N_eff / len(levels['min_bias_total_entropy']), label=f"DP {dp}")

    plt.title('Effective Sample Size / Total Events by Design Point')
    plt.xlabel('Design Point')
    plt.ylabel('Effective Sample Size / Total Events')
    plt.legend()
    plt.grid()
    plt.show()
