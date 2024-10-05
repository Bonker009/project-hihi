import matplotlib.pyplot as plt
import pandas as pd

# Mock SQL query result
data = [('Italian', 25), ('American', 22), ('Indian', 20), ('Mexican', 19), ('Chinese', 14)]

# Convert data to a pandas dataframe
df = pd.DataFrame(data, columns=['cuisine_type', 'count']).set_index('cuisine_type')

# Create the plot
plt.figure(figsize=(8, 6))
plt.bar(df.index, df['count'], color='skyblue')
plt.xlabel('Cuisine Type')
plt.ylabel('Count')
plt.title('Top 5 Cuisine Types by Count')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('top_5_cuisine_types.png')

# Show the plot
plt.show()