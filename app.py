
from flask import Flask, request, jsonify, send_file
import os
from flask_cors import CORS  # Import CORS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import joblib

# Load datasets
genomic_data_path = './enhanced_aquatic_population_data.csv'
environmental_data_path = './environmental_dataset.csv'

genomic_data = pd.read_csv(genomic_data_path)
environmental_data = pd.read_csv(environmental_data_path)
# Merge datasets on Population_ID
combined_data = pd.merge(genomic_data, environmental_data, on='Population_ID')

# Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load pre-trained models
try:
    trait_model = joblib.load("trait_frequency_model.pkl")
    trait_model_B = joblib.load("trait_frequency_model_B.pkl")  # Load model for Trait_Frequency_B
    selection_model = joblib.load("selection_pressure_model.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# Define feature columns
features_trait = ['Population_Size', 'Migration_Rate', 'Temperature_C', 'Selection_Pressure']
features_selection = ['Temperature_C', 'pH_Level', 'Environment', 'Dissolved_Oxygen']

@app.route('/predict_trait_frequency', methods=['POST'])
def predict_trait_frequency():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        # Predict Trait_Frequency_A
        prediction = trait_model.predict(input_df[features_trait])[0]
        return jsonify({"Predicted_Trait_Frequency_A": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_trait_frequency_b', methods=['POST'])
def predict_trait_frequency_b():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        # Predict Trait_Frequency_B
        prediction = trait_model_B.predict(input_df[features_trait])[0]
        return jsonify({"Predicted_Trait_Frequency_B": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_selection_pressure', methods=['POST'])
def predict_selection_pressure():
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])
    try:
        prediction = selection_model.predict(input_df[features_selection])[0]
        return jsonify({"Predicted_Selection_Pressure": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/simulate_scenario', methods=['POST'])
def simulate_scenario_api():
    input_data = request.get_json()
    variable = input_data.get("variable")
    values = input_data.get("values")
    fixed_features = input_data.get("fixed_features")

    try:
        scenarios = []
        for value in values:
            scenario = fixed_features.copy()
            scenario[variable] = value
            scenarios.append(scenario)
        scenario_df = pd.DataFrame(scenarios)
        predictions = trait_model.predict(scenario_df[features_trait])
        scenario_df['Predicted_Trait_Frequency_A'] = predictions
        return jsonify(scenario_df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/recommend_breeding_program', methods=['GET'])
def recommend_breeding_program():
    try:
        # Ensure no division by zero or NaN values
        valid_data = combined_data[(combined_data['Fitness_B'] > 0) & combined_data['Fitness_A'].notnull() & combined_data['Fitness_B'].notnull()]

        # Compute optimal conditions per population
        optimal_conditions = valid_data.groupby('Population_ID').apply(
            lambda x: x.loc[(x['Fitness_A'] / x['Fitness_B']).idxmax()]
        )

        # Reset index and select relevant columns
        optimal_conditions = optimal_conditions.reset_index(drop=True)
        response = optimal_conditions[['Population_ID', 'Temperature_C', 'pH_Level', 'Fitness_A', 'Fitness_B']]

        return jsonify(response.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})
    


@app.route('/heatmap_trait_distribution', methods=['GET'])
def heatmap_trait_distribution_combined():
    try:
        # Check if Trait_B exists
        trait_B_exists = 'Trait_Frequency_B' in combined_data.columns

        # Generate pivot table for Trait A
        heatmap_data_A = combined_data.pivot_table(
            values='Trait_Frequency_A',
            index='Generation',
            columns='Population_ID',
            aggfunc='mean'
        )

        # Generate pivot table for Trait B (if it exists)
        heatmap_data_B = combined_data.pivot_table(
            values='Trait_Frequency_B',
            index='Generation',
            columns='Population_ID',
            aggfunc='mean'
        ) if trait_B_exists else None

        # Create a figure for combined heatmaps
        fig, axes = plt.subplots(2 if trait_B_exists else 1, 1, figsize=(10, 12))
        
        # Heatmap for Trait A
        sns.heatmap(heatmap_data_A, cmap='coolwarm', annot=True, fmt=".2f", ax=axes[0])
        axes[0].set_title("Trait Frequency Heatmap (Trait A)")

        # Heatmap for Trait B (if exists)
        if trait_B_exists:
            sns.heatmap(heatmap_data_B, cmap='coolwarm', annot=True, fmt=".2f", ax=axes[1])
            axes[1].set_title("Trait Frequency Heatmap (Trait B)")

        # Save the combined heatmaps to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # Return the combined heatmap image
        return send_file(buf, mimetype='image/png', download_name='combined_trait_heatmaps.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/lineplot_simulation', methods=['GET'])
def lineplot_simulation():
    try:
        # Simulate trait frequency over generations
        simulated_data = combined_data.copy()
        simulated_data['Generation'] = simulated_data['Generation'] % 10

        # Simulate Trait A
        simulated_data['Trait_Frequency_A'] *= (1 + 0.01 * simulated_data['Generation'])

        # Simulate Trait B if it exists
        if 'Trait_Frequency_B' in simulated_data.columns:
            simulated_data['Trait_Frequency_B'] *= (1 + 0.015 * simulated_data['Generation'])

        # Create a single figure with subplots
        fig, axes = plt.subplots(2 if 'Trait_Frequency_B' in simulated_data.columns else 1, 1, figsize=(12, 12))

        # Plot Trait A
        sns.lineplot(data=simulated_data, x='Generation', y='Trait_Frequency_A', hue='Population_ID', ax=axes[0] if 'Trait_Frequency_B' in simulated_data.columns else axes)
        axes[0].set_title("Simulated Trait Frequency Over Generations (Trait A)")
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Trait Frequency A")
        axes[0].legend(title='Population ID')

        # Plot Trait B if available
        if 'Trait_Frequency_B' in simulated_data.columns:
            sns.lineplot(data=simulated_data, x='Generation', y='Trait_Frequency_B', hue='Population_ID', ax=axes[1])
            axes[1].set_title("Simulated Trait Frequency Over Generations (Trait B)")
            axes[1].set_xlabel("Generation")
            axes[1].set_ylabel("Trait Frequency B")
            axes[1].legend(title='Population ID')

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the combined plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # Return the combined image
        return send_file(buf, mimetype='image/png', download_name='combined_lineplot_simulation.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/barplot_population_fitness', methods=['GET'])
def barplot_population_fitness():
    try:
        # Aggregate fitness data for bar plot
        fitness_data = combined_data.groupby('Population_ID').agg({
            'Fitness_A': 'mean',
            'Fitness_B': 'mean'
        }).reset_index()

        # Generate bar plot
        plt.figure(figsize=(10, 6))
        fitness_data.plot(x='Population_ID', kind='bar', stacked=True, figsize=(10, 6))
        plt.title("Population Fitness Comparison")
        plt.xlabel("Population ID")
        plt.ylabel("Fitness Levels")
        plt.legend(["Fitness A", "Fitness B"], title="Trait Fitness")

        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(buf, mimetype='image/png', download_name='barplot_population_fitness.png')
    except Exception as e:
        return jsonify({"error": str(e)})


# Start the Flask server
if __name__ == '__main__':
    # Get the port from environment variable (Render will set this)
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=False)  # Bind to 0.0.0.0 for external access
