import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
INPUT_CSV_PATH = "regime_features_with_labels.csv"
LATENT_DIM = 4  # Increased Latent Dimension
EPOCHS = 100 # Number of epochs for training VAE (can be tuned)
BATCH_SIZE = 32 # Batch size for training (can be tuned)
N_SYNTHETIC_SAMPLES_PER_REGIME = 500 # Number of synthetic samples to generate per regime
VAE_BETA = 0.1 # Weight for KL loss term (Beta-VAE)

# --- VAE Model Definition ---

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a feature set."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(original_dim, latent_dim):
    encoder_inputs = layers.Input(shape=(original_dim,))
    # Adjust hidden layersarchitecture based on original_dim complexity
    h = layers.Dense(int(original_dim * 0.75) if original_dim * 0.75 > latent_dim else 64, activation='relu')(encoder_inputs) 
    h = layers.Dense(int(original_dim * 0.5) if original_dim * 0.5 > latent_dim else 32, activation='relu')(h)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def build_decoder(original_dim, latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    # Decoder architecture should mirror encoder's complexity
    h = layers.Dense(int(original_dim * 0.5) if original_dim * 0.5 > latent_dim else 32, activation='relu')(latent_inputs)
    h = layers.Dense(int(original_dim * 0.75) if original_dim * 0.75 > latent_dim else 64, activation='relu')(h)
    decoder_outputs = layers.Dense(original_dim, activation='sigmoid')(h) # Sigmoid if features are normalized to [0,1]
                                                                         # Or linear if features are standardized (mean 0, std 1)
                                                                         # We will use StandardScaler, so linear is better.
                                                                         # However, for stability with VAEs, often a final tanh or sigmoid is used
                                                                         # and data is scaled to [-1,1] or [0,1]. Let's start with linear for now.
                                                                         # Revisit if generation quality is poor.
    # For standardized data (mean 0, std 1), a linear activation is typically better for the output layer.
    decoder_outputs_linear = layers.Dense(original_dim, activation=None)(h) 
    decoder = Model(latent_inputs, decoder_outputs_linear, name="decoder")
    return decoder

class VAE(Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta # Store beta
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # Ensure data is float32 for stability with some operations
        data = tf.cast(data, tf.float32) 
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Optional: Add graph-compatible checks for numerics
            data = tf.debugging.check_numerics(data, "Input data check")
            z_mean = tf.debugging.check_numerics(z_mean, "z_mean check")
            z_log_var = tf.debugging.check_numerics(z_log_var, "z_log_var check")
            z = tf.debugging.check_numerics(z, "z sampling check")
            reconstruction = tf.debugging.check_numerics(reconstruction, "Reconstruction check")

            reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = reconstruction_loss_fn(data, reconstruction)
            reconstruction_loss = tf.debugging.check_numerics(reconstruction_loss, "Reconstruction loss check")
            
            # Clamp z_log_var 
            z_log_var = tf.clip_by_value(z_log_var, -20.0, 20.0) 

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = tf.debugging.check_numerics(kl_loss, "KL loss check")
            
            # Apply beta weight to KL loss
            total_loss = reconstruction_loss + self.beta * kl_loss 
            total_loss = tf.debugging.check_numerics(total_loss, "Total loss check")
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # Apply gradients only if they are not NaN/Inf (more robust)
        valid_grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_weights) if g is not None]
        # Apply check_numerics to gradients as well before clipping/applying
        finite_grads = [(tf.debugging.check_numerics(g, f"Gradient check for {v.name}"), v) for g,v in valid_grads_and_vars]
        # Clipping gradients can sometimes help stability
        clipped_grads_and_vars = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in finite_grads]
        
        # Check finite status AFTER check_numerics (which would raise error if non-finite)
        # We might not strictly need the len check anymore if check_numerics is used, but keep for safety?
        # Or rely solely on check_numerics to halt execution on bad gradients.
        # Let's rely on check_numerics raising an error if grads are bad.
        self.optimizer.apply_gradients(clipped_grads_and_vars)

        # We can assume losses are finite if check_numerics didn't raise an error
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
            
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

# --- Main Script Logic Placeholder ---
def main():
    # 1. Load Data
    try:
        data_df = pd.read_csv(INPUT_CSV_PATH, index_col='Date', parse_dates=True)
        print(f"Successfully loaded data from {INPUT_CSV_PATH}. Shape: {data_df.shape}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {INPUT_CSV_PATH}. Please ensure analyze_dow_data.py has run and saved the file.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if 'Market_Regime' not in data_df.columns:
        print("Error: 'Market_Regime' column not found in the loaded data. Cannot proceed.")
        return

    feature_columns = [col for col in data_df.columns if col != 'Market_Regime']
    if not feature_columns:
        print("Error: No feature columns found (excluding Market_Regime). Cannot proceed.")
        return
    
    print(f"Feature columns for VAE: {feature_columns}")
    original_dim = len(feature_columns)

    all_synthetic_data = {}
    trained_vaes = {}
    scalers_per_regime = {}

    # 2. Train VAE for each regime
    for regime_id in sorted(data_df['Market_Regime'].unique()):
        print(f"\n--- Processing Market Regime {int(regime_id)} ---")
        regime_data = data_df[data_df['Market_Regime'] == regime_id][feature_columns].copy()

        # Print original stats for Regime 0 before any modification
        if regime_id == 0:
            print("\nOriginal Data Stats for Regime 0 (Before Any Handling):")
            try:
                print(regime_data.describe())
            except Exception as desc_e:
                print(f"Could not describe data for Regime 0: {desc_e}")
        
        # --- Handle NaNs BEFORE scaling ---
        initial_nan_counts = regime_data.isnull().sum()
        if initial_nan_counts.sum() > 0:
            print(f"\nHandling NaNs in original data for Regime {int(regime_id)} (using ffill then bfill):")
            print(initial_nan_counts[initial_nan_counts > 0])
            regime_data.ffill(inplace=True)
            regime_data.bfill(inplace=True)
            # Check if any NaNs remain after filling
            remaining_nans = regime_data.isnull().sum().sum()
            if remaining_nans > 0:
                print(f"Warning: {remaining_nans} NaNs remain even after ffill/bfill for Regime {int(regime_id)}. This might indicate leading/trailing NaNs across all columns. Dropping these rows.")
                regime_data.dropna(inplace=True)
        # --- End NaN Handling ---

        if regime_data.shape[0] < BATCH_SIZE * 2: 
            print(f"Regime {int(regime_id)} has only {regime_data.shape[0]} samples after NaN handling. Too few for training. Skipping.")
            continue
        
        # a. Scale data for this regime
        scaler = StandardScaler()
        try:
            scaled_regime_data = scaler.fit_transform(regime_data)
            scalers_per_regime[regime_id] = scaler 
            print(f"Data scaled for Regime {int(regime_id)}. Shape: {scaled_regime_data.shape}")
        except ValueError as scale_e:
            print(f"Error scaling data for Regime {int(regime_id)}: {scale_e}. Skipping.")
            continue 

        # Check for NaNs/Infs AFTER scaling (should ideally not happen now)
        if np.any(np.isnan(scaled_regime_data)) or np.any(np.isinf(scaled_regime_data)):
            print(f"Error: NaNs or Infs detected in scaled_regime_data for Regime {int(regime_id)} AFTER scaling. Skipping.")
            continue 

        # b. Build VAE components
        encoder = build_encoder(original_dim, LATENT_DIM)
        decoder = build_decoder(original_dim, LATENT_DIM)
        vae = VAE(encoder, decoder, beta=VAE_BETA)
        
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)) # Adjusted learning rate
        
        print(f"Training VAE for Regime {int(regime_id)}...")
        # Consider adding a validation split for early stopping in a more advanced setup
        history = vae.fit(scaled_regime_data, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=True) # Set verbose=1 to see epoch progress
        trained_vaes[regime_id] = vae
        print(f"VAE training complete for Regime {int(regime_id)}.")

        # --- Plot Training History ---
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Total Loss')
            plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
            plt.plot(history.history['kl_loss'], label='KL Loss')
            plt.title(f'VAE Training History for Regime {int(regime_id)}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.ylim(bottom=0) # Start y-axis from 0 for better visualization of loss magnitudes
            plt.show()
        except Exception as hist_e:
            print(f"Error plotting training history for regime {regime_id}: {hist_e}")
        # --- End Plot Training History ---

        # c. Generate synthetic data
        print(f"Generating {N_SYNTHETIC_SAMPLES_PER_REGIME} synthetic samples for Regime {int(regime_id)}...")
        # Sample from standard normal distribution in latent space
        random_latent_vectors = np.random.normal(size=(N_SYNTHETIC_SAMPLES_PER_REGIME, LATENT_DIM))
        synthetic_scaled_data = decoder.predict(random_latent_vectors)
        
        # d. Inverse transform to original scale
        synthetic_original_scale_data = scaler.inverse_transform(synthetic_scaled_data)
        synthetic_df = pd.DataFrame(synthetic_original_scale_data, columns=feature_columns)
        all_synthetic_data[regime_id] = synthetic_df
        print(f"Synthetic data generated for Regime {int(regime_id)}. Shape: {synthetic_df.shape}")

    # 3. Analyze/Compare Synthetic Data
    print("\n--- Analysis of Synthetic Data ---")
    for regime_id, synthetic_df in all_synthetic_data.items():
        print(f"\n--- Regime {int(regime_id)} --- ")
        original_regime_data = data_df[data_df['Market_Regime'] == regime_id][feature_columns]
        
        if original_regime_data.empty or synthetic_df.empty:
             print("Original or synthetic data is empty, skipping analysis for this regime.")
             continue

        # a. Statistical Comparison (Mean and Std Dev)
        print("\nStatistical Comparison:")
        stats_compare_df = pd.DataFrame({
            'Original Mean': original_regime_data.mean(),
            'Synthetic Mean': synthetic_df.mean(),
            'Original Std Dev': original_regime_data.std(),
            'Synthetic Std Dev': synthetic_df.std()
        })
        print(stats_compare_df)

        # b. Distribution Comparison (KDE Plots for all features)
        print("\nGenerating Distribution Plots (KDE)...")
        n_features = len(feature_columns)
        n_cols = 2 # Arrange plots in 2 columns
        n_rows = (n_features + n_cols - 1) // n_cols
        fig_kde, axes_kde = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 4))
        axes_kde = axes_kde.flatten() # Flatten the axes array for easy iteration

        for i, col in enumerate(feature_columns):
            ax = axes_kde[i]
            try:
                sns.kdeplot(original_regime_data[col], ax=ax, label="Original", fill=True, alpha=.5, linewidth=0)
                sns.kdeplot(synthetic_df[col], ax=ax, label="Synthetic", fill=True, alpha=.5, linewidth=0)
                ax.set_title(f"Distribution of {col}")
                ax.legend()
            except Exception as plot_e:
                print(f"  Error plotting KDE for {col}: {plot_e}")
                ax.set_title(f"Error plotting {col}")
        
        # Hide any unused subplots if n_features is odd
        for j in range(i + 1, len(axes_kde)):
            fig_kde.delaxes(axes_kde[j])
            
        fig_kde.suptitle(f"Feature Distributions for Regime {int(regime_id)}", fontsize=16, y=1.02)
        fig_kde.tight_layout()
        plt.show()

        # c. Correlation Comparison (Heatmaps)
        print("\nGenerating Correlation Heatmaps...")
        try:
            original_corr = original_regime_data.corr()
            synthetic_corr = synthetic_df.corr()
            
            fig_corr, axes_corr = plt.subplots(1, 2, figsize=(12, 5))
            
            sns.heatmap(original_corr, ax=axes_corr[0], annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
            axes_corr[0].set_title('Original Data Correlation')
            
            sns.heatmap(synthetic_corr, ax=axes_corr[1], annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
            axes_corr[1].set_title('Synthetic Data Correlation')
            
            fig_corr.suptitle(f"Feature Correlations for Regime {int(regime_id)}", fontsize=16, y=1.0)
            fig_corr.tight_layout()
            plt.show()
        except Exception as corr_e:
            print(f"  Error generating correlation heatmaps: {corr_e}")

        # d. Optional: Pair Plot (can be uncommented if desired)
        # print("\nGenerating Pair Plots (might take time)...")
        # try:
        #     # Add a source column for differentiation in pairplot
        #     plot_pair_df_orig = original_regime_data.copy()
        #     plot_pair_df_orig['Source'] = 'Original'
        #     plot_pair_df_synth = synthetic_df.copy()
        #     plot_pair_df_synth['Source'] = 'Synthetic'
        #     combined_pair_df = pd.concat([plot_pair_df_orig, plot_pair_df_synth], ignore_index=True)
        #     
        #     sns.pairplot(combined_pair_df, hue='Source', diag_kind='kde')
        #     plt.suptitle(f"Pair Plot for Regime {int(regime_id)}", y=1.02)
        #     plt.show()
        # except Exception as pair_e:
        #     print(f"  Error generating pair plot: {pair_e}")

    print("\n--- Analysis of Synthetic Data Complete ---")

    # --- ADDED: Save all synthetic data to a single CSV for use in analyze_dow_data.py ---
    if all_synthetic_data:
        combined_synthetic_df_list = []
        for regime_id, synthetic_df in all_synthetic_data.items():
            # Add a column for the original regime ID these synthetic samples belong to
            temp_df = synthetic_df.copy()
            temp_df['Original_Market_Regime'] = regime_id 
            combined_synthetic_df_list.append(temp_df)
        
        if combined_synthetic_df_list:
            final_combined_synthetic_df = pd.concat(combined_synthetic_df_list, ignore_index=True)
            output_synthetic_csv = "all_regimes_synthetic_features.csv"
            try:
                final_combined_synthetic_df.to_csv(output_synthetic_csv, index=False)
                print(f"\nSuccessfully saved all combined synthetic features to: {output_synthetic_csv}")
                print(f"Shape of combined synthetic data: {final_combined_synthetic_df.shape}")
            except Exception as e:
                print(f"\nError saving combined synthetic features to CSV: {e}")
        else:
            print("\nNo synthetic data was generated to save.")
    else:
        print("\nNo synthetic data available to save.")
    # --- END ADDED CODE ---

    print("\n--- generate_regime_data.py script finished ---")

if __name__ == "__main__":
    main() 