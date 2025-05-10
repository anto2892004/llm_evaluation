
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


results_df = pd.read_csv("results/model_evaluation_results.csv")

results_melted = results_df.melt(id_vars=["Model"], 
                                  var_name="Metric", 
                                  value_name="Score")


sns.set(style="whitegrid", font_scale=1.2)


plt.figure(figsize=(12, 6))
ax = sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric", palette="tab10")


plt.title("Comparison of Model Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Metric")
plt.tight_layout()


plt.savefig("model_comparison_chart.png")
plt.show()
