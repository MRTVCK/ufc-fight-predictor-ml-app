import os
import pandas as pd
import numpy as np
import gradio as gr
import joblib
import warnings
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from math import pi
from io import BytesIO


warnings.filterwarnings("ignore")

# === CONFIGURATION ===
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_csv = os.path.join(script_dir, "data", "large_dataset.csv")

# === DATA PREP ===
def load_and_prepare_data():
    df = pd.read_csv(dataset_csv)
    df = df[df['winner'].isin(['Red', 'Blue'])]
    df['stance_diff'] = (df['r_stance'] != df['b_stance']).astype(int)
    df = df.sort_values(by='event_name', ascending=True).reset_index(drop=True)
    df.dropna(subset=[
        'age_diff', 'reach_diff', 'height_diff', 'weight_diff',
        'wins_total_diff', 'losses_total_diff',
        'sig_str_acc_total_diff', 'td_acc_total_diff',
        'SLpM_total_diff', 'SApM_total_diff',
        'str_def_total_diff', 'td_def_total_diff',
        'stance_diff', 'sub_avg_diff', 'td_avg_diff'
    ], inplace=True)
    return df

# === FEATURE ENGINEERING ===
def build_features(df):
    features = [
        'age_diff', 'reach_diff', 'height_diff', 'weight_diff',
        'wins_total_diff', 'losses_total_diff',
        'sig_str_acc_total_diff', 'td_acc_total_diff',
        'SLpM_total_diff', 'SApM_total_diff',
        'str_def_total_diff', 'td_def_total_diff',
        'stance_diff', 'sub_avg_diff', 'td_avg_diff'
    ]
    X = df[features].copy()
    y = LabelEncoder().fit_transform(df['winner'])
    return X, y, features

# === MODEL TRAINING ===
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("\n=== Model Evaluation ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))
    print("\nClassification Report:\n", classification_report(y_test, model.predict(X_test), target_names=['Blue', 'Red']))
    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validated Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
    joblib.dump(model, os.path.join(script_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(script_dir, 'scaler.pkl'))
    return model, scaler

# === EXTRA MODELS ===
def train_extra_models(df):
    features = [
        'age_diff', 'reach_diff', 'height_diff', 'weight_diff',
        'wins_total_diff', 'losses_total_diff',
        'sig_str_acc_total_diff', 'td_acc_total_diff',
        'SLpM_total_diff', 'SApM_total_diff',
        'str_def_total_diff', 'td_def_total_diff', 'stance_diff',
        'sub_avg_diff', 'td_avg_diff'
    ]
    X = df[features]
    label_encoder = LabelEncoder()
    y_method = label_encoder.fit_transform(df['method'])
    joblib.dump(label_encoder, os.path.join(script_dir, 'method_encoder.pkl'))
    outputs = {
        'method_model.pkl': y_method,
        'sig_strike_model_red.pkl': df['r_sig_str'],
        'sig_strike_model_blue.pkl': df['b_sig_str'],
        'takedown_model_red.pkl': df['r_td'],
        'takedown_model_blue.pkl': df['b_td'],
        'round_model.pkl': df['finish_round']
    }
    for filename, y in outputs.items():
        model = RandomForestClassifier() if 'method' in filename else RandomForestRegressor()
        model.fit(X, y)
        joblib.dump(model, os.path.join(script_dir, filename))

def load_extra_models():
    models = {}
    filenames = [
        'method_model.pkl', 'sig_strike_model_red.pkl', 'sig_strike_model_blue.pkl',
        'takedown_model_red.pkl', 'takedown_model_blue.pkl', 'round_model.pkl'
    ]
    for f in filenames:
        models[f] = joblib.load(os.path.join(script_dir, f))
    models['method_encoder'] = joblib.load(os.path.join(script_dir, 'method_encoder.pkl'))
    return models

# === GET STATS ===
def get_latest_fighter_stats(name, df):
    fights = df[(df['r_fighter'] == name) | (df['b_fighter'] == name)]
    if fights.empty:
        return None
    fights = fights.sort_values(by='event_name', ascending=False)
    row = fights.iloc[0]
    prefix = 'r_' if row['r_fighter'] == name else 'b_'
    return {
        'Name': name,
        'Age': row[f'{prefix}age'],
        'Reach': row[f'{prefix}reach'],
        'Height': row[f'{prefix}height'],
        'Weight': row[f'{prefix}weight'],
        'Wins': row[f'{prefix}wins_total'],
        'Losses': row[f'{prefix}losses_total'],
        'SigStrPct': row[f'{prefix}sig_str_acc_total'],
        'TDPct': row[f'{prefix}td_acc_total'],
        'SLpM': row.get(f'{prefix}SLpM_total', 0),
        'SApM': row.get(f'{prefix}SApM_total', 0),
        'str_def': row.get(f'{prefix}str_def_total', 0),
        'td_def': row.get(f'{prefix}td_def_total', 0),
        'Stance': row.get(f'{prefix}stance', 'Unknown'),
        'sub_avg': row.get(f'{prefix}sub_avg', 0),
        'td_avg': row.get(f'{prefix}td_avg', 0)
    }

# === PLOTTING ===
def generate_radar_chart(r_stats, b_stats, red_name, blue_name):
    labels = ['Age', 'Reach', 'Height', 'Weight', 'Wins', 'Losses', 'SigStrPct', 'TDPct']
    r_values = [
        r_stats['Age'], r_stats['Reach'], r_stats['Height'], r_stats['Weight'],
        r_stats['Wins'], r_stats['Losses'], r_stats['SigStrPct'], r_stats['TDPct']
    ]
    b_values = [
        b_stats['Age'], b_stats['Reach'], b_stats['Height'], b_stats['Weight'],
        b_stats['Wins'], b_stats['Losses'], b_stats['SigStrPct'], b_stats['TDPct']
    ]

    # Normalize values for fair comparison on radar
    max_vals = np.maximum(r_values, b_values)
    r_values_norm = [r / m if m != 0 else 0 for r, m in zip(r_values, max_vals)]
    b_values_norm = [b / m if m != 0 else 0 for b, m in zip(b_values, max_vals)]

    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    r_values_norm += r_values_norm[:1]
    b_values_norm += b_values_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, r_values_norm, linewidth=2, label=red_name, color='red')
    ax.fill(angles, r_values_norm, alpha=0.25, color='red')
    ax.plot(angles, b_values_norm, linewidth=2, label=blue_name, color='blue')
    ax.fill(angles, b_values_norm, alpha=0.25, color='blue')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.set_title("Fighter Stat Comparison (Radar)", fontsize=13)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_bar_chart(labels, values1, values2, title, label1, label2):
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax.bar(x - 0.2, values1, width=0.4, label=label1)
    ax.bar(x + 0.2, values2, width=0.4, label=label2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def plot_pie_chart(probs, labels):
    fig, ax = plt.subplots()
    ax.pie(probs, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title("Win Probability")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# === LAUNCH GUI ===
def launch_gradio_gui(model, scaler, df, extra_models):
    fighter_names = sorted(set(df['r_fighter']).union(set(df['b_fighter'])))
    weight_classes = sorted(df['weight_class'].unique())

    def get_fighters_by_weight(weight_class):
        fights = df[df['weight_class'] == weight_class]
        names = set(fights['r_fighter']).union(set(fights['b_fighter']))
        return sorted(list(names))

    def update_fighters(weight_class, mode):
        all_fighters = sorted(set(df['r_fighter']).union(set(df['b_fighter'])))
        if mode == "Freestyle":
            return (
                gr.update(choices=all_fighters, interactive=True),
                gr.update(choices=all_fighters, interactive=True),
                gr.update(interactive=False)  # Disable dropdown
            )
        else:
            fighters = get_fighters_by_weight(weight_class)
            return (
                gr.update(choices=fighters, interactive=True),
                gr.update(choices=fighters, interactive=True),
                gr.update(interactive=True)
            )

    def lookup_fighter(name):
        stats = get_latest_fighter_stats(name, df)
        if not stats:
            return f"❌ No stats found for {name}"
        lines = [f"📋 Stats for {name}:"]
        for k, v in stats.items():
            if k != "Name":
                lines.append(f"{k}: {v}")
        return "\n".join(lines)


    # === Predict Fight Mode ===
    def predict_fight(red, blue, mode, rounds):
        r_stats, b_stats = get_latest_fighter_stats(red, df), get_latest_fighter_stats(blue, df)
        if not r_stats or not b_stats:
            return "❌ Missing fighter data.", None, None, None, None

        features = np.array([
            r_stats['Age'] - b_stats['Age'],
            r_stats['Reach'] - b_stats['Reach'],
            r_stats['Height'] - b_stats['Height'],
            r_stats['Weight'] - b_stats['Weight'],
            r_stats['Wins'] - b_stats['Wins'],
            r_stats['Losses'] - b_stats['Losses'],
            r_stats['SigStrPct'] - b_stats['SigStrPct'],
            r_stats['TDPct'] - b_stats['TDPct'],
            r_stats['SLpM'] - b_stats['SLpM'],
            r_stats['SApM'] - b_stats['SApM'],
            r_stats['str_def'] - b_stats['str_def'],
            r_stats['td_def'] - b_stats['td_def'],
            int(r_stats['Stance'] != b_stats['Stance']),
            r_stats['sub_avg'] - b_stats['sub_avg'],
            r_stats['td_avg'] - b_stats['td_avg']
        ]).reshape(1, -1)

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        probs = model.predict_proba(scaled)[0]
        winner = red if prediction == 1 else blue

        method_idx = extra_models['method_model.pkl'].predict(features)[0]
        method_label = extra_models['method_encoder'].inverse_transform([method_idx])[0]

        ssr = int(extra_models['sig_strike_model_red.pkl'].predict(features)[0])
        ssb = int(extra_models['sig_strike_model_blue.pkl'].predict(features)[0])
        tdr = int(extra_models['takedown_model_red.pkl'].predict(features)[0])
        tdb = int(extra_models['takedown_model_blue.pkl'].predict(features)[0])

        rnd = int(extra_models['round_model.pkl'].predict(features)[0])
        if mode == "Realistic" and "Decision" in method_label:
            rnd = int(rounds)

        result = f"🍊 Predicted Winner: {winner}\n"
        result += f"🔒 Confidence: {probs[prediction]*100:.2f}%\n"
        result += f"💥 Method: {method_label}\n"
        result += f"📊 Strikes — {red}: {ssr}, {blue}: {ssb}\n"
        result += f"🤼 Takedowns — {red}: {tdr}, {blue}: {tdb}\n"
        result += f"⏱ Finish Round: {rnd}\n"

        if probs[prediction] >= 0.8:
            result += "🎯 High confidence finish"
        elif "Decision" in method_label:
            result += "📏 Likely going the distance"
        else:
            result += "⚠️ Moderate chance of finish"

        radar_chart = generate_radar_chart(r_stats, b_stats, red, blue)
        strike_chart = plot_bar_chart(["Sig Strikes"], [ssr], [ssb], "Significant Strikes", red, blue)
        td_chart = plot_bar_chart(["Takedowns"], [tdr], [tdb], "Takedowns", red, blue)
        pie_chart = plot_pie_chart([probs[1], probs[0]], [red, blue])

        return result, radar_chart, strike_chart, td_chart, pie_chart
    

# === Fight Prediction Tab ===
# === Start building UI ===
    with gr.Blocks() as app:
        gr.Markdown("## 🥋 UFC FIGHT PREDICTOR\n**Invented by: Destin Tucker**")

        with gr.Tab("Look up a Fighter"):
            fighter_dropdown = gr.Dropdown(choices=fighter_names, label="Select Fighter")
            fighter_output = gr.Textbox(label="Fighter Stats")
            fighter_dropdown.change(fn=lookup_fighter, inputs=fighter_dropdown, outputs=fighter_output)

        with gr.Tab("Predict a Fight"):
            with gr.Row():
                mode_radio = gr.Radio(["Realistic", "Freestyle"], label="Mode", value="Realistic")
                weight_class_dropdown = gr.Dropdown(choices=weight_classes, label="Select Weight Class")

            with gr.Row():
                red_dropdown = gr.Dropdown(choices=[], label="Red Corner")
                blue_dropdown = gr.Dropdown(choices=[], label="Blue Corner")

            rounds_slider = gr.Slider(minimum=1, maximum=5, step=1, label="Rounds", value=3)

            gr.Markdown("👉 **Choose 'Realistic' to simulate a UFC-eligible fight.**\n\n🎮 **Or choose 'Freestyle' to match any two fighters, no matter the weight class!**")

            predict_btn = gr.Button("Predict Fight")
            prediction_output = gr.Textbox(label="Prediction Results",
                    lines=20,          # initial visible height (rows)
                    max_lines=40,      # lets it grow before showing a scrollbar
                    show_copy_button=True,
                    autoscroll=False)
            radar_output = gr.Image(label="Radar Chart")
            strike_output = gr.Image(label="Significant Strikes")
            td_output = gr.Image(label="Takedowns")
            pie_output = gr.Image(label="Win Probability")

            weight_class_dropdown.change(
                fn=update_fighters,
                inputs=[weight_class_dropdown, mode_radio],
                outputs=[red_dropdown, blue_dropdown, weight_class_dropdown]
            )
            mode_radio.change(
                fn=update_fighters,
                inputs=[weight_class_dropdown, mode_radio],
                outputs=[red_dropdown, blue_dropdown, weight_class_dropdown]
            )
            predict_btn.click(
                fn=predict_fight,
                inputs=[red_dropdown, blue_dropdown, mode_radio, rounds_slider],
                outputs=[prediction_output, radar_output, strike_output, td_output, pie_output]
            )

    app.launch()

# === MAIN ===
def main():
    df = load_and_prepare_data()
    X, y, features = build_features(df)
    model, scaler = train_model(X, y)
    train_extra_models(df)
    extra_models = load_extra_models()
    print("✅ All models trained and ready.")
    launch_gradio_gui(model, scaler, df, extra_models)

if __name__ == "__main__":
    main()
