# -------------------------------------------------------------------------
# AUTHOR: Joshua Furman
# FILENAME: retail_dataset.csv
# SPECIFICATION: Finds strong association rules in a retail dataset and outputs them based on specified thresholds.
# FOR: CS 4210- Assignment #5
# TIME SPENT: 3 hrs
# -----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

# Find the unique items and store them in a set
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

# Remove nan (empty) values
itemset.remove(np.nan)

# Convert the dataset to meet the requirements of the apriori module
encoded_vals = []
for index, row in df.iterrows():
    labels = {}
    for item in itemset:
        if item in row.values:
            labels[item] = 1
        else:
            labels[item] = 0
    encoded_vals.append(labels)

# Create a dataframe from the populated list of dictionaries
ohe_df = pd.DataFrame(encoded_vals)

# Apply the apriori algorithm
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# Print the results
for index, rule in rules.iterrows():
    antecedents = ', '.join(list(rule['antecedents']))
    consequents = ', '.join(list(rule['consequents']))
    support = rule['support']
    confidence = rule['confidence']
    support_count = ohe_df[rule['consequents']].sum(axis=1).value_counts().get(1, 0)
    prior = support_count / len(encoded_vals)
    gain_in_confidence = (confidence - prior) / prior * 100

    print(antecedents + " -> " + consequents)
    print("Support: " + str(support))
    print("Confidence: " + str(confidence))
    print("Prior: " + str(prior))
    print("Gain in Confidence: " + str(gain_in_confidence))
    print()

# Plot support vs confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
