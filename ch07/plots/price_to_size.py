import matplotlib.pyplot as plt
import pandas as pd

data_path = "/home/willieloea/Laboratory/datasets/foml_datasets/tab7_1.csv"
df = pd.read_csv(data_path)

x = df["SIZE"]
y = df["RENTAL PRICE"]

# 2. Create the figure and axis
fig, ax = plt.subplots(figsize=(5, 5))

# In the book, points are drawn as open circles with black outlines
ax.scatter(x, y, facecolors="none", edgecolors="black", s=25, linewidths=1.2)

# 3. Configure axes limits and labels to match Figure 7.1(a)
ax.set_xlabel("Size", fontsize=10)
ax.set_ylabel("Rental Price", fontsize=10)

ax.set_xlim(450, 1050)
ax.set_ylim(300, 650)

# Optional: set exact ticks as seen in the book
ax.set_xticks([500, 600, 700, 800, 900, 1000])
ax.set_yticks([350, 400, 450, 500, 550, 600])

plt.title("(a)", y=-0.25, fontsize=10)
plt.tight_layout()

# 4. Display or save
# plt.show()
# or to save:
plt.savefig("fig7_1a.png", dpi=300)

