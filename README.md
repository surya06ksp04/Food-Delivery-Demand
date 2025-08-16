# Food Delivery Demand Prediction

Predict peak-hour demand, map delivery zones, and suggest optimal driver allocation for food delivery platforms (Swiggy/Zomato/UberEats or simulated data).  
Designed for **intermediate Python** learners—clean code, comments, and ready to push to GitHub.

## Features
- **Data simulation** if you don't have real data (orders + weather).
- **Preprocessing**: time features, weather joins, zone creation via K-Means clustering.
- **Modeling**: Random Forest to predict orders per zone-hour.
- **Peak hours**: Detect & plot busiest times.
- **Driver allocation**: Heuristic to recommend drivers per zone-hour.
- **Map visualization**: Interactive Folium map of zones colored by expected demand.

## Repo Structure
```
food-delivery-demand/
├── data/
│   └── simulated_orders.csv              # created by simulate_data.py
├── models/
│   └── demand_model.pkl                  # saved by train_model.py
├── outputs/
│   ├── peak_hours.png                    # created by predict_and_allocate.py
│   └── demand_map.html                   # created by visualize_map.py
├── src/
│   ├── simulate_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict_and_allocate.py
│   ├── visualize_map.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Quickstart
1. **Create & activate env (optional)**
   ```bash
   python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```
2. **Install deps**
   ```bash
   pip install -r requirements.txt
   ```
3. **Simulate data** (skip if you have your own)
   ```bash
   python src/simulate_data.py
   ```
4. **Preprocess & build zones**
   ```bash
   python src/preprocess.py
   ```
5. **Train model**
   ```bash
   python src/train_model.py
   ```
6. **Predict & allocate drivers**
   ```bash
   python src/predict_and_allocate.py
   ```
7. **Make map**
   ```bash
   python src/visualize_map.py
   ```

## Using Real Datasets
- Replace `data/simulated_orders.csv` with your CSV containing at least:
  - `order_id, order_ts, lat, lon, items, prep_time_min, temperature_c, is_rain`
- Optional: include `zone_id` per order. If missing, the pipeline will create zones with K-Means.
- Update column names in `src/utils.py` if yours differ.

## Driver Allocation Heuristic
Drivers needed per zone-hour:
```
drivers = ceil( expected_orders * avg_service_time_min / 60 )
```
- `avg_service_time_min` defaults to **20 min** (prep + pickup + travel); change via CLI.

## Interview Talking Points
- Built an **end-to-end pipeline** (simulation → features → model → optimization).
- Used **time-series style** aggregations (zone-hour) and **geo clustering** for zones.
- Delivered **actionable ops insights** (peak-hour staffing & coverage map).

## License
MIT
