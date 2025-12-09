import json
import pandas as pd

with open('personal_first_500_data.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Income > 150000 AND 'No Investments'
df['Eligible'] = df.apply(
    lambda row: "Yes" if row['Income'] > 150000 and row['Mutual Funds or Investments'] == 'No Investments' else "No",
    axis=1
)

# Convert DataFrame back to list of dicts
updated_data = df.to_dict(orient='records')

# Save to new JSON file
with open('labeled_data.json', 'w') as f:
    json.dump(updated_data, f, indent=4)

print("Labeled data saved to 'labeled_data.json'")
