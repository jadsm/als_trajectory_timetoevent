import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

def get_km_values(durations, events, groups, alpha=0.05):
    """
    Extract Kaplan-Meier survival curves data for different groups.
    
    Parameters:
    -----------
    durations : array-like
        Array of observed durations
    events : array-like
        Array of event indicators (1 if event occurred, 0 if censored)
    groups : array-like
        Array of group labels for stratified analysis
    alpha : float, optional
        Significance level for confidence intervals (default: 0.05)
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames with survival curve data for each group:
        - timeline: time points
        - survival_function: survival probabilities
        - confidence_intervals: upper and lower CI bounds
        - at_risk: number of subjects at risk
        - observed: number of events observed
        - censored: number of censored observations
    """
    # Create a DataFrame for easier handling
    df = pd.DataFrame({
        'duration': durations,
        'event': events,
        'group': groups
    })
    
    # Initialize dictionary to store results
    results = {}
    
    # Analyze each group
    for group in df['group'].unique():
        mask = (df['group'] == group)
        group_data = df[mask]
        
        # Fit KM model
        kmf = KaplanMeierFitter()
        kmf.fit(
            group_data['duration'],
            group_data['event'],
            label=f'Group {group}',
            alpha=alpha
        )
        
        # Get confidence intervals
        ci_df = kmf.confidence_interval_
        
        # Calculate at-risk counts, observed events, and censored counts at each time point
        at_risk = []
        observed = []
        censored = []
        
        for t in kmf.timeline:
            at_risk.append(sum(group_data['duration'] >= t))
            obs = sum((group_data['duration'] == t) & (group_data['event'] == 1))
            cens = sum((group_data['duration'] == t) & (group_data['event'] == 0))
            observed.append(obs)
            censored.append(cens)
        
        # Compile results into a DataFrame
        results[group] = pd.DataFrame({
            'timeline': kmf.timeline,
            'survival_function': kmf.survival_function_.values.flatten(),
            'confidence_interval_lower': ci_df.iloc[:, 0].values,
            'confidence_interval_upper': ci_df.iloc[:, 1].values,
            'at_risk': at_risk,
            'observed': observed,
            'censored': censored
        })
        
        # Add summary statistics
        results[f'group_{group}_stats'] = {
            'median_survival': kmf.median_survival_time_,
            'events_observed': sum(group_data['event']),
            'total_subjects': len(group_data)
        }
    
    return results

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 200

    # Create two groups with different survival distributions
    durations = np.concatenate([
        np.random.exponential(50, n_samples),
        np.random.exponential(70, n_samples)
    ])
    events = np.random.binomial(1, 0.7, len(durations))
    groups = np.repeat([0, 1], n_samples)

    # Get KM curve data
    results = get_km_values(durations, events, groups)

    # Example of accessing the data
    for group, data in results.items():
        if isinstance(data, pd.DataFrame):
            print(f"\nFirst few rows of survival data for {group}:")
            print(data.head())
        else:
            print(f"\nSummary statistics for {group}:")
            for stat, value in data.items():
                print(f"{stat}: {value:.2f}")