"""
Smart Kitchen — Flask Backend + HTML Frontend
Run: python app.py
Opens at: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import json
import os
import tensorflow as tf

app = Flask(__name__)

# ── Load all model files ─────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

print("⏳ Loading model and files...")
model      = tf.keras.models.load_model(os.path.join(BASE, "best_model.keras"))
le_dish    = joblib.load(os.path.join(BASE, "le_dish.pkl"))
le_ing     = joblib.load(os.path.join(BASE, "le_ingredient.pkl"))
le_unit    = joblib.load(os.path.join(BASE, "le_unit.pkl"))
le_cat     = joblib.load(os.path.join(BASE, "le_category.pkl"))
scaler_X   = joblib.load(os.path.join(BASE, "scaler_X.pkl"))
scaler_y   = joblib.load(os.path.join(BASE, "scaler_y.pkl"))
lookup     = json.load(open(os.path.join(BASE, "dish_ingredient_lookup.json")))
model_info = json.load(open(os.path.join(BASE, "model_info.json")))
print("✅ All files loaded!")

NGO_THRESHOLD = 500  # grams — if waste > this, trigger NGO alert

# ── Helper: predict quantity for one ingredient ──────────────
def predict_qty(dish_name, ingredient, num_customers, item):
    try:
        features = np.array([[
            le_dish.transform([dish_name])[0],
            le_ing.transform([ingredient])[0],
            num_customers,
            item["shelf_life_days"],
            item["perishability_score"],
            item["wastage_factor_percent"],
            le_unit.transform([item["unit"]])[0],
            le_cat.transform([item["category"]])[0],
        ]], dtype=np.float32)

        scaled  = scaler_X.transform(features).reshape(1, 1, -1)
        pred_sc = model.predict(scaled, verbose=0)[0][0]
        pred    = scaler_y.inverse_transform([[pred_sc]])[0][0]
        return round(max(float(pred), 0), 2)
    except Exception:
        return round(item["quantity_per_person_grams"] * num_customers, 2)


# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    dishes = sorted(list(lookup.keys()))
    return render_template("index.html", dishes=dishes, model_info=model_info)


@app.route("/api/ingredients/<dish_name>")
def get_ingredients(dish_name):
    if dish_name not in lookup:
        return jsonify({"error": "Dish not found"}), 404
    return jsonify({"ingredients": lookup[dish_name]})


@app.route("/api/predict", methods=["POST"])
def predict():
    data         = request.json
    dish_name    = data.get("dish_name")
    num_customers = int(data.get("num_customers", 1))

    if dish_name not in lookup:
        return jsonify({"error": "Dish not found"}), 404

    results = []
    for item in lookup[dish_name]:
        qty = predict_qty(dish_name, item["ingredient"], num_customers, item)
        results.append({
            "ingredient"            : item["ingredient"],
            "required_grams"        : qty,
            "required_kg"           : round(qty / 1000, 3),
            "unit"                  : item["unit"],
            "category"              : item["category"],
            "shelf_life_days"       : item["shelf_life_days"],
            "perishability_score"   : item["perishability_score"],
            "wastage_factor_percent": item["wastage_factor_percent"],
        })

    return jsonify({"dish": dish_name, "num_customers": num_customers, "ingredients": results})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data          = request.json
    dish_name     = data.get("dish_name")
    num_customers = int(data.get("num_customers", 1))
    stock_map     = {s["ingredient"]: float(s["stock_grams"]) for s in data.get("stock", [])}

    if dish_name not in lookup:
        return jsonify({"error": "Dish not found"}), 404

    analysis  = []
    ngo_items = []

    for item in lookup[dish_name]:
        ing      = item["ingredient"]
        required = predict_qty(dish_name, ing, num_customers, item)
        in_stock = stock_map.get(ing, 0.0)
        diff     = in_stock - required

        if diff >= 0:
            status   = "EXCESS"
            excess   = round(diff, 2)
            shortage = 0.0
            waste    = round(excess * item["wastage_factor_percent"] / 100, 2)
            ngo_flag = waste > NGO_THRESHOLD
        else:
            status   = "SHORTAGE"
            excess   = 0.0
            shortage = round(abs(diff), 2)
            waste    = 0.0
            ngo_flag = False

        if ngo_flag:
            ngo_items.append({
                "ingredient"           : ing,
                "excess_grams"         : excess,
                "predicted_waste_grams": waste,
                "shelf_life_days"      : item["shelf_life_days"]
            })

        analysis.append({
            "ingredient"            : ing,
            "unit"                  : item["unit"],
            "in_stock_grams"        : in_stock,
            "required_grams"        : required,
            "status"                : status,
            "excess_grams"          : excess,
            "shortage_grams"        : shortage,
            "predicted_waste_grams" : waste,
            "perishability_score"   : item["perishability_score"],
            "ngo_alert"             : ngo_flag
        })

    return jsonify({
        "dish"         : dish_name,
        "num_customers": num_customers,
        "analysis"     : analysis,
        "ngo_required" : len(ngo_items) > 0,
        "ngo_items"    : ngo_items
    })


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🍽️  Smart Kitchen is running!")
    print("="*50)
    print("👉  Open this in your browser:")
    print("    http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
